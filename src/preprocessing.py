import datasets
import numpy as np
import pandas as pd
import torch
from tokenizers import ByteLevelBPETokenizer
from torch import FloatTensor
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer

from config.device import DEVICE
from feature_extractor import fingerprint_features


def preprocessing_pipeline(path_train=None, path_val=None,
                           path_test=None):
    
    def get_features_smile(data):
        return fingerprint_features(data).ToList()

    dataset = {}
    if path_train:
        df = pd.read_csv(path_train)
        mean_label = df["P1"].mean()
        dataset["train"] = {
            "inputs": FloatTensor(np.array(df["smiles"].apply(get_features_smile).values.tolist())).to(DEVICE),
            "labels": FloatTensor(df["P1"]).to(DEVICE),
            "weights": FloatTensor([mean_label if y == 0 else 1 - mean_label for y in df["P1"]]).to(DEVICE)
        }
    if path_val:
        df = pd.read_csv(path_val)
        mean_label = df["P1"].mean()
        dataset["val"] = {
            "inputs": FloatTensor(np.array(df["smiles"].apply(get_features_smile).values.tolist())).to(DEVICE),
            "labels": FloatTensor(df["P1"]).to(DEVICE),
            "weights": FloatTensor([mean_label if y == 0 else 1 - mean_label for y in df["P1"]]).to(DEVICE)
        }
    if path_test:
        df = pd.read_csv(path_test)
        mean_label = df["P1"].mean()
        dataset["test"] = {
            "inputs": FloatTensor(np.array(df["smiles"].apply(get_features_smile).values.tolist())).to(DEVICE)
        }

    return dataset

def train_tokenizer(path_text: str, path_output: str, vocab_size: int=52000,
                    min_frequency: int=2) -> int:
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=path_text, vocab_size=vocab_size, min_frequency=min_frequency,
                    special_tokens=["<unk>","<mask>", "<pad>", "<s>", "</s>"])
    # Save files to disk
    tokenizer.save_model(path_output)

def preprocess_smile_lm(path_text: str, path_tokenizer: str):
    tokenizer = RobertaTokenizer.from_pretrained(path_tokenizer)
    df_text = pd.read_csv(path_text)
    inputs_tokenized = tokenizer(df_text["smiles"].values.tolist(),
                                 max_length=128, padding='max_length',
                                 truncation=True, return_tensors='pt')

    def masked_lm(tensor):
        rand = torch.rand(tensor.shape)
        mask_arr = (rand < 0.15) * (tensor > 4)
        for i in range(tensor.shape[0]):
            selection = torch.flatten(mask_arr[i].nonzero()).tolist()
            tensor[i, selection] = 4
        return tensor
    
    labels = inputs_tokenized.input_ids
    mask = inputs_tokenized.attention_mask
    input_ids = masked_lm(inputs_tokenized.input_ids.detach().clone())
    encodings = {
        'input_ids': input_ids,
        'mask': mask,
        'labels': labels
    }

    class Dataset(datasets.dataset_dict.DatasetDict):
        def __init__(self, encodings):
            self.encodings = encodings
            
        def __len__(self):
            return self.encodings["input_ids"].shape[0]
        
        def __getitem__(self, i):
            return {key: tensor[i] for key, tensor in self.encodings.items()}

    dataset = Dataset(encodings)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    return dataloader, tokenizer.vocab_size

def preprocessing_lm_pipeline(path_text, path_tokenizer) -> tuple:
    """
    """
    train_tokenizer(path_text, path_tokenizer)
    dataloader, vocab_size = preprocess_smile_lm(path_text, path_tokenizer)
    return dataloader, vocab_size

def preprocessing_nlp_pipeline(path_tokenizer, path_train=None, path_val=None,
                               path_test=None, max_length=128):
    tokenizer = RobertaTokenizer.from_pretrained(path_tokenizer)
    paths = {}
    if path_train: paths["train"] = path_train
    if path_val: paths["val"] = path_val
    if path_test: paths["test"] = path_test
    if len(paths) == 0: return None, None, None
    
    dataset = datasets.load_dataset('csv', data_files=paths)

    def tokenize_function(data):
        return tokenizer(data["smiles"], max_length=max_length,
                         padding='max_length', truncation=True)

    tokenized_datasets = dataset.map(tokenize_function)
    tokenized_datasets = tokenized_datasets.remove_columns(['smiles', 'mol_id', 'attention_mask'])
    tokenized_datasets = tokenized_datasets.rename_column("P1", "labels")
    for key in paths.keys():
        tokenized_datasets[key].set_format("torch")

    dataloader = {}
    if "train" in tokenized_datasets:
        dataloader["train"] = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=8)
    if "val" in tokenized_datasets:
        dataloader["val"] = DataLoader(tokenized_datasets["val"], batch_size=8)
    if "test" in tokenized_datasets:
        dataloader["test"] = DataLoader(tokenized_datasets["test"], batch_size=8)

    return dataloader
