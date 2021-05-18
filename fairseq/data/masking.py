import math
from tqdm import tqdm
import numpy as np
import torch
from math import sqrt, log
from itertools import chain
import json
import random
from collections import defaultdict


class SpanMaskingScheme:
    def __init__(self, span_lower, span_upper, geometric_p, mask_idx, mask_ratio , pad_idx, tgt_dict , no_mask=False):
        self.lower = span_lower
        self.upper = span_upper
        self.mask_ratio = mask_ratio
        self.lens = list(range(self.lower, self.upper + 1))
        self.p = geometric_p
        self.len_distrib = [self.p * (1-self.p)**(i - self.lower) for i in range(self.lower, self.upper + 1)] if self.p >= 0 else None
        self.len_distrib = [x / (sum(self.len_distrib)) for x in self.len_distrib]
        self.mask_idx = mask_idx
        self.pad_idx = pad_idx
        self.tokens = tgt_dict.indices.values()
        self.no_mask = no_mask

    def mask(self, span_list, sentence):
        sent_length = len(span_list)
        mask_num = math.ceil(sent_length * self.mask_ratio)
        mask = set()
        i = 0

        while len(mask) < mask_num:            
            span_len = np.random.choice(self.lens, p=self.len_distrib)
            anchor = np.random.choice(sent_length)

            left, right = span_list[str(anchor)]

            for i in range(left, right) :
                if len(mask) >= mask_num:
                    break                
                mask.add(i)
            
            while len(mask) < mask_num and right < sent_length :
                if right >= sent_length :
                    break
                anchor = right
                left, right = span_list[str(anchor)]
                
                for i in range(left, right) :
                    if len(mask) >= mask_num:
                        break                
                    mask.add(i)        



        span_input = sentence.clone()
        if self.no_mask==True:
            span_output = sentence.clone()
        else:
            span_output = torch.full_like(sentence, self.pad_idx)

        for i in range( len(span_input) ): # bpe 토근 길이만큼 돈다. eos 마스킹 안걸림ㅎㅎ
            if i in mask:
                span_output[i] = sentence[i]
                rand = np.random.random()
                if rand < 0.8 :
                    span_input[i] = self.mask_idx
                elif rand < 0.9 :
                    span_input[i] = np.random.choice( list(self.tokens) )
        

        return span_input, span_output


class BertMaskingScheme: 
    def __init__(self, mask_idx, mask_ratio , pad_idx, tgt_dict , no_mask=False):
        self.mask_ratio = mask_ratio
        self.mask_idx = mask_idx
        self.pad_idx = pad_idx
        self.tokens = tgt_dict.indices.values()
        self.no_mask = no_mask        

    def mask(self, sentence):
        sent_length = len(sentence) - 1 # ignore eos
        mask_num = math.ceil(sent_length * self.mask_ratio)
        mask = np.random.choice(sent_length, mask_num, replace=False)

        span_input = sentence.clone()
        if self.no_mask==True:
            span_output = sentence.clone()
        else:
            span_output = torch.full_like(sentence, self.pad_idx)

        for i in mask :
            span_output[i] = sentence[i]
            rand = np.random.random()
            if rand < 0.8 :
                span_input[i] = self.mask_idx
            elif rand < 0.9 :
                span_input[i] = np.random.choice( list(self.tokens) )
        
        return span_input , span_output
 