import os
import re
import pandas as pd 
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.autograd import  Variable 
from torch.utils.data import DataLoader, Dataset
from AttentionTransformer.TrainClassificationTransformer import * 
from AttentionTransformer.ClassificationDataset import *
from AttentionTransformer.utilities import count_model_parameters
import pickle 
from tqdm import tqdm_notebook, tqdm, trange, tnrange 
import torch.optim as optim
from tokenizers import BertWordPieceTokenizer
import logging


from TwoClassHeadClassificationTransformer import *
from ClassificationDatasetFromDict import *


logging.basicConfig(filename='classification_fp32_training.log', filemode='a', level=logging.INFO, format = '{asctime} {filename} {message}', style="{")

SEED = 3007
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

def load_pickle(filepath):
    with open(filepath, 'rb') as fp:
        return pickle.load(fp)


tokenizer = BertWordPieceTokenizer('../data/bert-word-piece-custom-wikitext-vocab-10k-vocab.txt', lowercase = True, strip_accents = True)


data = load_pickle('../data/tokenized_questions_classes_subclasses_dict.pkl')


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

model = model.to('cuda')

model = model.train()

param_txt = f'''
    Total trainiable model parameters: {count_model_parameters(model) / 1e6} Million
'''
print(param_txt)
logging.info(param_txt)


dataset = ClassificationDatasetDict(data, seq_len)
dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers=num_workers, pin_memory=True)


optimizer = optim.Adam(model.parameters(), lr=1e-4, betas = (0.50, 0.90))
clip = 2
EPOCHS = 1000
EPOCHS_START = 1
loss_, acc_ = [], []

# acc_.append(dict_['accuracy'])
for epoch in trange(EPOCHS_START, EPOCHS+EPOCHS_START):

    total_loss = 0
    total_loss_class, n_label_total_class, n_label_correct_class = 0, 0, 0
    total_loss_subclass, n_label_total_subclass, n_label_correct_subclass = 0, 0, 0

    for ix, batch in enumerate(tqdm(dataloader, desc='DataLoader')):

        src_seq = batch['source_seq'].to('cuda')
        class_label = batch['class'].to('cuda')
        subclass_label = batch['subclass'].to('cuda')

        src_seq, class_label, subclass_label = Variable(src_seq), Variable(class_label), Variable(subclass_label)

        optimizer.zero_grad()

        pred = model(src_seq)

        # loss, n_correct, n_label = cal_performance(pred, gold.long(), target_pad_id)

        loss_class, n_correct_class, n_total_class = classification_performance(pred[0], class_label)
        loss_subclass, n_correct_subclass, n_total_subclass = classification_performance(pred[1], subclass_label)

        loss = loss_class + loss_subclass

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        n_label_total_class += n_total_class
        n_label_correct_class += n_correct_class

        n_label_total_subclass += n_total_subclass
        n_label_correct_subclass += n_correct_subclass

        total_loss += loss.item()


    loss_per_label = total_loss / n_label_total_class
    accuracy_class = n_label_correct_class / n_label_total_class
    accuracy_subclass = n_label_correct_subclass / n_label_total_subclass
    accuracy_mean = np.mean((accuracy_class, accuracy_subclass))


    if epoch == 1:
        torch.save(model, '../models/initial_classification_model.pt')
        torch.save({
            "model_dict": model.state_dict(),
            "optimizer_dict": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss_per_label,
            "accuracy_class": accuracy_class,
            "accuracy_subclass": accuracy_subclass,
            "accuracy_mean": accuracy_mean,
            "loss_class": loss_class, 
            "loss_subclass": loss_subclass
        }, '../models/initial_classification_model_state_dict.pth')


    if len(acc_) > 0 and accuracy_mean > max(acc_):

        torch.save(model, '../models/classification_model_best.pt')
        torch.save({
            "model_dict": model.state_dict(),
            "optimizer_dict": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss_per_label,
            "accuracy_class": accuracy_class,
            "accuracy_subclass": accuracy_subclass,
            "accuracy_mean": accuracy_mean,
            "loss_class": loss_class, 
            "loss_subclass": loss_subclass
        }, '../models/classification_model_state_dict_best.pth')

    acc_.append(accuracy_mean)
    loss_.append(loss_per_label)

    text = f'EPOCH: {epoch:.5f} | Loss: {loss_per_label:.5f} | Accuracy Class: {accuracy_class:.5f} | Accuracy Subclass: {accuracy_subclass:.5f}'

    print(text)

    logging.info(f'EPOCH: {epoch} | Loss: {loss_per_label} | Accuracy Class: {accuracy_class} | Accuracy Subclass: {accuracy_subclass}')

