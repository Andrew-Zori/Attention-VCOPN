import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model import *

deepcopy = copy.deepcopy

def make_model(generator, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    # Basic Components
    attn = MultiHeadAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    
    encoder = Encoder(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), dropout), generator, N)
 
    model = encoder

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


class Batch(object):
    def __init__(self, src, tgt = None, pad = 0):
        # src.size(): (batch_size, max_len)
        self.src = src
        
        self.tgt = tgt
        
        self.ntokens = self.tgt.size(0)

def run_epoch(data_iter, model, loss_compute):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src)
        
        # out size: (batch_size, pos-1, d_model)
        loss = loss_compute(out, batch.tgt, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        # tokens is used to sum all the generated tokens every 50 batches
        tokens += batch.ntokens
        if i % 50 == 1:
            print('*' * 20)
            elapsed = time.time() - start + 0.00000001
            print("Epoch Step: %d  Loss: %f   Tokens per Sec: %f" 
                    %(i, (loss/batch.ntokens).item(), tokens / elapsed))
        start = time.time()
        tokens = 0
        
    return total_loss / total_tokens

class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        tmp_a = step**(-0.5)
        tmp_b = step * self.warmup**(-1.5)
        return self.factor * (self.model_size)**(-0.5) * min(tmp_a, tmp_b)

    
    
    
class Generator(nn.Module):
    # Generator = Linear Layer(Projection)

    def __init__(self, dimen_model, dimen_vocab):
        # dimen_model: the dimension of decoder output
        # dimen_vocab: the dimension of vocabulary 
        super(Generator, self).__init__()
        self.projection_layer = nn.Linear(dimen_model, dimen_vocab)

    def forward(self, x):
        # input.size(): (batch_size, max_len, d_model)
        # output.size(): (batch_size, max_len, d_vocab)
        return self.projection_layer(x)
       
