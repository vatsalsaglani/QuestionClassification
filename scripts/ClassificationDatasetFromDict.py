import os
import re 
import torch 
import torch.nn as nn
from torch.utils.data import Dataset 

SEED = 3007
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

class ClassificationDatasetDict(Dataset):

    def __init__(self, dataDict, seq_len):

        super(ClassificationDatasetDict, self).__init__()

        self.dataDict = dataDict 
        self.seq_len = seq_len 

    def pad_sequences(self, seq):

        if len(seq) > self.seq_len:

            seq = seq[:self.seq_len]

        op = torch.tensor(seq)
        tnsr = torch.zeros((self.seq_len))
        tnsr[:op.size(0)] = op
        return tnsr.float()

    def __len__(self):
        return len(self.dataDict)

    def __getitem__(self, ix):

        ixDict = self.dataDict[ix]
        seq = self.pad_sequences(ixDict['question-tokens'])
        cls_ = ixDict['question-class']
        subcls_ = ixDict['question-subclass']

        return {
            "source_seq": seq.float(), 
            "class": cls_, 
            "subclass": subcls_
        }