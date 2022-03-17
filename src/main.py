import argparse

from preprocessing import (preprocessing_lm_pipeline,
                           preprocessing_nlp_pipeline, preprocessing_pipeline)
from train import train_lm_pipeline, train_nlp_pipeline, train_pipeline

parser = argparse.ArgumentParser(description='Train =models for drug property prediction.')

parser.add_argument('--path_train', dest='path_train', 
                    nargs='?', default='../data/train.csv',
                    help='Path to the train set file')
parser.add_argument('--path_val', dest='path_val', 
                    nargs='?', default='../data/val.csv',
                    help='Path to the validation set file')
parser.add_argument('--path_text', dest='path_text', 
                    nargs='?', default='../data/smiles_string.csv',
                    help='Path to the file containing smile strings')
parser.add_argument('--path_tokenizer', dest='path_tokenizer', 
                    nargs='?', default='../models/',
                    help='Path to the folder containing tokenizer files')
parser.add_argument('--model_save_path', dest='model_save_path', 
                    nargs='?', default='../models/checkpoint.pth',
                    help='Path to the folder where to save models')
parser.add_argument('--lm_model_save_path', dest='lm_model_path', 
                    nargs='?', default='../models/lm-checkpoint.pth',
                    help='Path to the folder contaning language model file')
parser.add_argument('--path_lm_model', dest='path_lm_model', 
                    nargs='?', default='../models/',
                    help='Path to the folder contaning language model file')
parser.add_argument('--model_name', dest='model_name', 
                    nargs='?', default='classic',
                    help='Which model to choose for training (classic: train classic MLP, \
                    lm: train a language model for smiles string, nlp: train with a NLP model \
                    using the language model)')

args = parser.parse_args()

if __name__ == '__main__':
    if args.model_name == 'classic':
        # TRAIN a Multi Layer Perceptron clasfication model  
        dataset_preprocessed = preprocessing_pipeline(args.path_train, args.path_val)
        a = train_pipeline(dataset_preprocessed, args.model_save_path)
    elif args.model_name == 'lm':
        # TRAIN a language model (Roberta) needed for the NLP classification task
        dataloader, vocab_size = preprocessing_lm_pipeline(args.path_text, args.path_tokenizer)
        train_lm_pipeline(dataloader, vocab_size, args.model_save_path)
    elif args.model_name == 'nlp':
        # TRAIN a classification model by finetuning the language model on smile training set
        dataloader = preprocessing_nlp_pipeline(args.path_tokenizer, args.path_train, args.path_val)
        train_nlp_pipeline(dataloader, args.path_lm_model, args.model_save_path)
