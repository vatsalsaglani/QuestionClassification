import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np 
import os 
import re 
from AttentionTransformer.Encoder import * 
from AttentionTransformer.Decoder import * 


class OneEncoderTwoDecoderTransformer(nn.Module):

    def __init__(
        self, vocab_size, pad_id, CLS_label_id, emb_dim = 512, dim_model = 512, dim_inner = 2048,
        layers = 6, heads = 8, dim_key = 64, dim_value = 64, dropout = 0.1, num_pos = 200
    ):

        super(OneEncoderTwoDecoderTransformer, self).__init__()

        self.pad_id = pad_id 
        self.encoder = Encoder(
            vocab_size, emb_dim, layers, heads, dim_key, dim_value, dim_model, dim_inner, pad_id, dropout = dropout, num_pos = num_pos
        )

        self.decoder1 = Decoder(
            vocab_size, emb_dim, layers, heads, dim_key, dim_value, dim_model, dim_inner, pad_id, dropout = dropout, num_pos = num_pos
        )

        self.decoder2 = Decoder(
            vocab_size, emb_dim, layers, heads, dim_key, dim_value, dim_model, dim_inner, pad_id, dropout = dropout, num_pos = num_pos
        )

        self.decoder1heads = nn.Linear(dim_model, 6)

        self.decoder2heads = nn.Linear(dim_model, 47)

        for parameter in self.parameters():

            if parameter.dim() > 1:

                nn.init.xavier_uniform_(parameter)

        assert dim_model == emb_dim, f'Dimensions of all the module objects must be same'

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

    def get_decoder2_target(self, labels):

        tnsr = labels.float()
        return tnsr.unsqueeze(1)

    def forward(self, source_seq, classlabels):

        b, l = source_seq.size()
        targetdec1 = self.make_target_seq(b).to(source_seq.device)
        source_mask = self.get_pad_mask(source_seq, self.pad_id)
        targetdec1_mask = self.get_pad_mask(targetdec1, self.pad_id) & self.get_subsequent_mask(targetdec1)

        targetdec2 = self.get_decoder2_target(classlabels)
        targetdec2_mask = self.get_pad_mask(targetdec2, self.pad_id) & self.get_subsequent_mask(targetdec2)
        

        encoder_output = self.encoder(source_seq, source_mask)
        decoder_output_1 = self.decoder1(
            targetdec1, targetdec1_mask, encoder_output, source_mask
        )
        decoder_output_2 = self.decoder2(
            targetdec2, targetdec2_mask, encoder_output, source_mask
        )

        decoder_output_1 = decoder_output_1.view(decoder_output_1.size(0), -1)
        decoder_output_2 = decoder_output_2.view(decoder_output_2.size(0), -1)

        classheads1 = self.decoder1heads(decoder_output_1)
        classheads2 = self.decoder2heads(decoder_output_2)

        return classheads1, classheads2
