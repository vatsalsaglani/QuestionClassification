import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np 
import os 
import re 
from AttentionTransformer.Encoder import * 
from AttentionTransformer.Decoder import * 


class TwoClassHeadClassificationTransformer(nn.Module):


    def __init__(
        self, vocab_size, pad_id, CLS_label_id, num_class_heads, lst_num_cat_in_classes, emb_dim = 512, dim_model = 512, dim_inner = 2048,
        layers = 6, heads = 8, dim_key = 64, dim_value = 64, dropout = 0.1, num_pos = 200
    ):

        super(TwoClassHeadClassificationTransformer, self).__init__()

        self.pad_id = pad_id 

        self.encoder = Encoder(
            vocab_size, emb_dim, layers, heads, dim_key, dim_value, dim_model, dim_inner, pad_id, dropout = dropout, num_pos = num_pos
        )

        self.decoder = Decoder(
            vocab_size, emb_dim, layers, heads, dim_key, dim_value, dim_model, dim_inner, pad_id, dropout = dropout, num_pos = num_pos
        )

        # self.target_word_projection = nn.Linear(dim_model, num_classes, bias = False)

        self.class_heads = nn.ModuleList([nn.Linear(dim_model, lst_num_cat_in_classes[ix], bias=False) for ix in range(num_class_heads)])

        # for ix in range(num_class_heads):

        #     self.class_heads.append(nn.Linear(dim_model, lst_num_cat_in_classes[ix], bias = False))

        for parameter in self.parameters():

            if parameter.dim() > 1:

                nn.init.xavier_uniform_(parameter)

        assert dim_model == emb_dim, f'Dimensions of all the moduel outputs must be the same'

        self.x_logit_scale = 1

        self.cls_label_id = CLS_label_id

    def get_pad_mask(self, sequence, pad_id):

        return (sequence != pad_id).unsqueeze(-2)

    def get_subsequent_mask(self, sequence):

        batch_size, seq_length = sequence.size() 

        subsequent_mask = (
            1 - torch.triu(
                torch.ones((1, seq_length, seq_length), device=sequence.device), diagonal = 1
            )
        ).bool()

        return subsequent_mask

    def make_target_seq(self, batch_size):

        trg_tnsr = torch.zeros((batch_size, 1))
        trg_tnsr[trg_tnsr == 0] = self.cls_label_id
        return trg_tnsr.float()

    def forward(self, source_seq):

        

        b, l = source_seq.size()

        target_seq = self.make_target_seq(b).to(source_seq.device)

        source_mask = self.get_pad_mask(source_seq, self.pad_id)
        target_mask = self.get_pad_mask(target_seq, self.pad_id) & self.get_subsequent_mask(target_seq)

        encoder_output = self.encoder(source_seq, source_mask)
        decoder_output = self.decoder(target_seq, target_mask, encoder_output, source_mask)

        decoder_output = decoder_output.view(decoder_output.size(0), -1)

        # seq_logits = self.target_word_projection(decoder_output)

        class_seq_logits = [ch(decoder_output.to()) for ch in self.class_heads]

        return class_seq_logits
