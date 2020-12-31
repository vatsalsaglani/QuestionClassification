import os 
import re 
import pandas as pd
import numpy as np 
import pickle 
from tokenizers import BertWordPieceTokenizer
from TwoClassHeadClassificationTransformer import *
from ClassificationDatasetFromDict import *
import pickle 
import torch 
import torch.nn as nn 

SEED = 3007
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

global model, tokenizer, idx2class, idx2subclass, class2names, device, subclass2names

device = 'cpu'

def save_pickle(obj, filepath):
    with open(filepath, 'wb') as fp:
        pickle.dump(obj, fp)

def load_pickle(filepath):
    with open(filepath, 'rb') as fp:
        return pickle.load(fp)        

class2names = {
    "DESC": "DESCRIPTION",
    "ENTY": "ENTITY",
    "ABBR": "ABBREVIATION",
    "HUM": "HUMAN",
    "NUM": "NUMERIC",
    "LOC": "LOCATION"
}

class2names = load_pickle('../data/class2names.pkl')
subclass2names = load_pickle('../data/subclass2names.pkl')
print(subclass2names['other'])

idx2class = load_pickle('../data/idx2class.pkl')
idx2subclass = load_pickle('../data/idx2subclass.pkl')

tokenizer = BertWordPieceTokenizer('../data/bert-word-piece-custom-wikitext-vocab-10k-vocab.txt', lowercase = True, strip_accents = True)

vocab_size = tokenizer.get_vocab_size()
pad_id = 0
CLS_label_id = 2
num_class_heads = 2
lst_num_cat_in_classes = [6, 47]
seq_len = 100
batch_size = 256
num_workers = 3

model = TwoClassHeadClassificationTransformer(
    vocab_size=vocab_size, pad_id=pad_id, CLS_label_id=CLS_label_id,
    num_class_heads=num_class_heads, 
    lst_num_cat_in_classes=lst_num_cat_in_classes, num_pos=seq_len
)

model_dict = torch.load('../models/classification_model_state_dict_best.pth', map_location = device)
model.load_state_dict(model_dict['model_dict'])
model = model.to(device)
model = model.eval()

print(f'''
Model saved at: {model_dict['epoch']}
    Accuracy Class: {model_dict['accuracy_class']}
    Accuracy Subclass: {model_dict['accuracy_subclass']}
''')

def predictQuestionClassSubclass(text):

    tokens = torch.FloatTensor(tokenizer.encode(text).ids).unsqueeze(0).to(device)
    cls_, subcls = model(tokens)
    clsIdx = cls_.max(1)[-1].item()
    subclsIdx = subcls.max(1)[-1].item()

    return {
        "class": class2names[idx2class[clsIdx]],
        "subclass": subclass2names[idx2subclass[subclsIdx]]
    }

print(predictQuestionClassSubclass(input('Enter Question: ')))