# -*- coding:utf-8 -*-
import math
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import clones, attention, subsequent_mask

deepcopy = copy.deepcopy

class Encoder(nn.Module):
    def __init__(self, layer, generator, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.generator = generator

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        
        # self.self_attn is the Multi-Head Attention Layer
        self.self_attn = self_attn
        
        # self.feed_forward is PositionwiseFeedForward Layer
        self.feed_forward = feed_forward
        
        # SublayerConnection Layer
        self.sublayerconnections = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask = None):
        # self attention
        self_attention_x = self.sublayerconnections[0](x, lambda x: self.self_attn(x, x, x, mask = None))
        # feed forward
        feed_forward_x = self.sublayerconnections[1](self_attention_x, self.feed_forward)
        return feed_forward_x
    

"""
h is the number of the parallel attention layers. In paper, h=8
"""
class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        #print d_model, h
        assert d_model % h == 0
        
        # self.d_k is the reduced dimension of each parallel attention
        self.d_k = d_model // h
        self.h = h

        # self.linears is a list consists of 4 projection layers
        # self.linears[0]: Concat(W^Q_i), where i \in [1,...,h]. 
        # self.linears[1]: Concat(W^K_i), where i \in [1,...,h]. 
        # self.linears[2]: Concat(W^K_i), where i \in [1,...,h]. 
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # query.size() = key.size() = value.size() = (batch_size, max_len, d_model)
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)

        """
        do all the linear projection, after this operation
        query.size() = key.size() = value.size() = (batch_size, self.h, max_len, self.d_k)
        """
        query, key, value = \
                [linear(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2) for 
                        linear, x in zip(self.linears, (query, key, value))]
        """
        x.size(): (batch_size, h, max_len, d_v)
        self.attn.size(): (batch_size, h, max_len, d_v)
        """
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        """
        x.transpose(1,2).size(): (batch_size, max_len, h, d_v)
        the transpose operation is necessary
        x.size: (batch_size, max_len, h*d_v)
        """
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        # self.linears[-1] \in R^{hd_v \times d_{model}}
        return self.linears[-1](x)

"""
There are three types of Multi-Head Attention
1. Self-Attention in the Encoder
   Query: the output of the previous layer
   Key  : the output of the previous layer
   Value: the output of the previous layer

2. Self-Attention in the Decoder
   Query: the output of the previous layer
   Key  : the output of the previous layer
   Value: the output of the previous layer
   
3. Attention between Encoder and Decoder
   Query: the output of the previous decoder layer
   Key  : the output of the encoder
   Value: the output of the encoder
"""


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        w1_x = self.w_1(x)
        relu_x = F.relu(w1_x)
        dropout_x = self.dropout(relu_x)
        return self.w_2(dropout_x)
    
    
# Add & Norm step in the encoder    
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    
    # sublayer is a function defined by self attention or feed forward 
    def forward(self, x, sublayer):
        # Normalization
        norm_x = self.norm(x)
        # Sublayer function
        sublayer_x = sublayer(norm_x)
        # Dropout function
        dropout_x = self.dropout(x)
        # Residual connection
        return x + dropout_x
    
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # a_2, b_2 is trainable to scale means and std variance
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.ones(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class Generator(nn.Module):
    # Generator = Linear Layer(Projection) + Softmax Layer(for output the probability distribution of each words in vocubulary)

    def __init__(self, dimen_model, dimen_vocab):
        # dimen_model: the dimension of decoder output
        # dimen_vocab: the dimension of vocabulary 
        super(Generator, self).__init__()
        self.projection_layer = nn.Linear(dimen_model, dimen_vocab)

    def forward(self, x):
        # input.size(): (batch_size, max_len, d_model)
        # output.size(): (batch_size, max_len, d_vocab)
        return self.projection_layer(x)
        