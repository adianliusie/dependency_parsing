import torch
import numpy as np
import random

from typing import List, Tuple
from types import SimpleNamespace
from collections import Counter, defaultdict
from copy import deepcopy

from ..utils.general_utils import flatten

class Batcher():
    def __init__(self, max_len:int=None):
        self.device = torch.device('cpu')
        self.max_len = max_len
            
    def batches(self, data:List['Conversations'], bsz:int, shuffle:bool=False):
        convs = self._prep_convs(data)
        if shuffle: random.shuffle(convs)
        batches = [convs[i:i+bsz] for i in range(0,len(convs), bsz)]
        for batch in batches:
            yield self.batchify(batch)
    
    def batchify(self, batch:List[list]):
        """each input is input ids and mask for utt, + label"""
        conv_id, k, ids, labels = zip(*batch)  
        ids, mask = self._get_padded_ids(ids)
        labels = self._pad_seq(labels, pad_id=-100)
        return SimpleNamespace(
            conv_id=conv_id, k=k, ids=ids, mask=mask, labels=labels)

    ### PROCESSING METHODS ###################################################
    
    def _prep_convs(self, data:List['Conversations']):
        """ sequence classification input data preparation"""
        output = []
        for conv in data:
            conv_id = conv.conv_id
            
            #get all utterances in conv and labels
            ids   = [utt.ids for utt in conv.utts]
            
            ids_chnk = self._chunk(ids)
            for k, utt_ids in enumerate(ids_chnk):
                input_ids = flatten(utt_ids)
                labels = self._get_lm_labels(input_ids)
                output.append([conv_id, k, input_ids, labels])
        return output
    
    def _get_lm_labels(self, utts:List[list])->list:
        """ returns mask for all LM words (no speaker, utt sep)"""
        labels = utts[1:] + [-100]
        return labels
    
    def _chunk(self, ids:List[list]):
        """breaks the conv into chunks saller than max_len"""
        ids = deepcopy(ids)
        ids_chnk = []
        while len(flatten(ids)) > self.max_len:
            id_lens = np.cumsum([len(utt_ids) + 1 for utt_ids in ids])
            
            x = sum([i < self.max_len for i in id_lens]) #utts that fit in max_len
            if x>0:
                ids_chnk.append(ids[:x])
                ids = ids[x:]
            else:
                ids = ids[1:]
                
        if len(flatten(ids)) > min(self.max_len/2, 512):
            ids_chnk.append(ids)
    
        return ids_chnk
                
    ### Padding Util Methods ############################################################
    
    def _get_padded_ids(self, ids:list, pad_id=0)->("pad_ids", "pad_mask"):
        """ pads ids to be flat """
        max_len = max([len(x) for x in ids])
        padded_ids = [x + [pad_id]*(max_len-len(x)) for x in ids]
        mask = [[1]*len(x) + [0]*(max_len-len(x)) for x in ids]
        ids = torch.LongTensor(padded_ids).to(self.device)
        mask = torch.FloatTensor(mask).to(self.device)
        return ids, mask
    
    def _pad_seq(self, x:list, pad_id:int=0)->list:
        """pads input so can be put in a tensor"""
        max_len = max([len(i) for i in x])
        x_pad = [i + [pad_id]*(max_len-len(i)) for i in x]
        x_pad = torch.LongTensor(x_pad).to(self.device)
        return x_pad
    

    def __call__(self, data, bsz, shuffle=False):
        """routes the main method do the batches function"""
        return self.batches(data=data, bsz=bsz, shuffle=shuffle)
            
    def to(self, device:torch.device):
        """ sets the device of the batcher """
        self.device = device
    