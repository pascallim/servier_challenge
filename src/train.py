from sklearn.feature_selection import mutual_info_regression
from tokenizers import ByteLevelBPETokenizer
from torch import nn, optim, save
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (RobertaForMaskedLM, RobertaForSequenceClassification)

from config.device import DEVICE
from config.roberta_model import get_roberta_config
from model_smiles import MLP


def train_pipeline(dataset, model_save_path, epochs=10000, lr=0.005, epoch_eval=500,
                   train_epoch_log=100):
    # Initialize configuration for training
    model = MLP().to(DEVICE)
    criterion = nn.BCELoss()
    criterion_weights = nn.BCELoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=lr)
    min_val_loss = None
    # Train model
    for e in range(1, epochs+1):
        # Train
        model.train()
        optimizer.zero_grad()
        y_pred_train = model(dataset["train"]["inputs"])
        if "weights" in dataset["train"].keys():
            loss = criterion_weights(y_pred_train.squeeze(), dataset["train"]["labels"])
            loss = loss * dataset["train"]["weights"]
            loss = loss.mean()
        else:
            loss = criterion(y_pred_train.squeeze(), dataset["train"]["labels"])
        loss.backward()
        optimizer.step()
        if e % train_epoch_log == 0:
            print('Epoch {}: train loss: {}'.format(e, loss.item()))
        # Validation
        if "val" in dataset.keys():
            model.eval()
            if e % epoch_eval == 0:
                y_pred_val = model(dataset["val"]["inputs"])
                if "weights" in dataset["val"]:
                    loss_val = criterion_weights(y_pred_val.squeeze(), dataset["val"]["labels"])
                    loss_val = loss_val * dataset["val"]["weights"]
                    loss_val = loss_val.mean()
                else:
                    loss_val = criterion(y_pred_val.squeeze(), dataset["val"]["labels"])
                print('Epoch {}: val loss: {}'.format(e, loss_val.item()))
                # Save best checkpoint
                if min_val_loss is None: min_val_loss = loss_val
                if min_val_loss and min_val_loss > loss_val:
                    print('Epoch {}: Saving The Model'.format(e))
                    min_val_loss = loss_val
                    # Saving State Dict
                    save(model.state_dict(), model_save_path)

def train_lm_pipeline(dataloader: DataLoader, vocab_size: int,
                      model_save_path: str, epochs: int=2,
                      lr: float=1e-5):
    # Initialize configuration for training
    config = get_roberta_config(vocab_size)
    model = RobertaForMaskedLM(config)
    model = model.to(DEVICE)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    min_val_loss = None
    for epoch in range(1, epochs+1):
        loop = tqdm(dataloader, leave=True)
        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(DEVICE)
            mask = batch['mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            outputs = model(input_ids, attention_mask=mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            loop.set_description(f'Epoch: {epoch}')
            loop.set_postfix(loss=loss.item())
        if min_val_loss is None: min_val_loss = loss
        if min_val_loss and min_val_loss > loss:
            print('Epoch {}: Saving The Model'.format(epoch))
            min_val_loss = loss
            # Saving State Dict
            save(model.state_dict(), model_save_path)

def train_nlp_pipeline(dataloader, lm_model_path, model_save_path,
                       epochs=100, epoch_eval=10, lr=5e-5):
    # Initialize configuration for training
    model = RobertaForSequenceClassification.from_pretrained(lm_model_path)
    model = model.to(DEVICE)
    optimizer = SGD(model.parameters(), lr=lr)
    min_val_loss = None
    for epoch in range(1, epochs+1):
        # TRAIN
        model.train()
        loop = tqdm(dataloader["train"], leave=True)
        loop.set_description(f'Epoch: {epoch}')
        for batch in loop:
            optimizer.zero_grad()
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
        # EVAL
        model.eval()
        if epoch % epoch_eval == 0:
            loop_val = tqdm(dataloader["val"], leave=True)
            loop_val.set_description(f'Epoch: {epoch}')
            for batch_eval in loop_val:
                batch_eval = {k: v.to(DEVICE) for k, v in batch_eval.items()}
                outputs_eval = model(**batch_eval)
                loss_val = outputs_eval.loss
                loop_val.set_postfix(loss_eval=loss_val.item())
                # Save best checkpoint
                if min_val_loss is None: min_val_loss = loss_val
                if min_val_loss and min_val_loss > loss_val:
                    print('Epoch {}: Saving The Model'.format(epoch))
                    min_val_loss = loss_val
                    # Saving State Dict
                    save(model.state_dict(), model_save_path)
