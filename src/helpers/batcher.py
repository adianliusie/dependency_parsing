import torch
import numpy as np
import random

from typing import List, Tuple
from types import SimpleNamespace
from collections import Counter, defaultdict
from copy import deepcopy

from ..utils.general_utils import flatten

class Batcher():
    def __init__(self, formatting:str, max_len:int=None, C=None):
        self.device = torch.device('cpu')
        self.max_len = max_len
        self.formatting = formatting
        self.C = C
            
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
            spkrs = [utt.spkr_id for utt in conv.utts]
            
            ids_chnk, spkrs_chnk = self._chunk(ids, spkrs)
            for k, (ids, spkrs) in enumerate(zip(ids_chnk, spkrs_chnk)):
                spkrs_tok = self._tokenize_speakers(spkrs)
                utt_ids   = self._format_ids(ids, spkrs_tok)

                input_ids = flatten(utt_ids)
                labels = self._get_lm_labels(ids)
                assert len(input_ids) == len(labels)
                output.append([conv_id, k, input_ids, labels])

        return output
    
    def _format_ids(self, utts:List[list], spkrs_tok:list):    
        """ prepares the ids to include seperator/speaker information """
        if self.formatting == 'spkr_sep':
            utt_ids = [[s] + utt for utt, s in zip(utts, spkrs_tok)]
            
        if self.formatting == 'utt_sep':
            print('need to update the sep id')
            utt_ids = [[0] + utt for utt, s in zip(utts, spkrs_tok)]
        
        if self.formatting == 'merged':
            utt_ids = utts
           
        return utt_ids
    
    def _get_lm_labels(self, utts:List[list])->list:
        """ returns mask for all LM words (no speaker, utt sep)"""
        utts = deepcopy(utts)
        if self.formatting in ['spkr_sep', 'utt_sep']:
            utt_word_ids = [[-100] + utt for utt in utts]
        elif self.formatting == 'merged':
            utt_word_ids = utts
        
        utt_word_ids = flatten(utt_word_ids)
        labels = utt_word_ids[1:] + [-100]
        return labels
    
    def _chunk(self, ids:List[list], spkrs:list):
        """breaks the conv into chunks saller than max_len"""
        ids, spkrs = deepcopy(ids), deepcopy(spkrs)
        ids_chnk, spkrs_chnk = [], []
        while len(flatten(ids)) + len(spkrs) > self.max_len:
            id_lens = np.cumsum([len(utt_ids) + 1 for utt_ids in ids])
            
            x = sum([i < self.max_len for i in id_lens]) #utts that fit in max_len
            #x = x-randint(0,3) -> can add this line if I want to have extra stochasticity in later epochs
            if x>0:
                ids_chnk.append(ids[:x])
                spkrs_chnk.append(spkrs[:x])
                ids, spkrs = ids[x:], spkrs[x:]
            else:
                ids, spkrs = ids[1:], spkrs[1:]
                
        if len(flatten(ids)) > min(self.max_len/2, 512):
            ids_chnk.append(ids)
            spkrs_chnk.append(spkrs)
    
        return ids_chnk, spkrs_chnk
                
    def _tokenize_speakers(self, spkrs:List[int])->List[int]:
        """ returns tokenized speaker ids for the input list"""
        #Order the speakers by frequency
        max_num_spkrs = len(self.C.speaker_tokens)
        spkr_counts = Counter(spkrs)
        
        # Create random mappings from speakers to tokens
        com_speakers, _ = zip(*spkr_counts.most_common(max_num_spkrs-1))
        com_speakers = list(com_speakers)
        random.shuffle(com_speakers)
        spkr_tokens = self.C.speaker_tokens.copy()
        
        #assign token to each speaker (where the final token are all uncommon speakers)
        mappings = defaultdict(lambda: spkr_tokens[-1])
        for spkr, tok in zip(com_speakers, spkr_tokens[:-1]):
            mappings[spkr] = tok
        
        #tokenize the sequence 
        spkrs_tok = [mappings[spkr] for spkr in spkrs]
        return spkrs_tok
    
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
    