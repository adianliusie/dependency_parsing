import json
import re 
import os

from types import SimpleNamespace
from typing import List
from tqdm import tqdm 
from collections import Counter

from ..utils.general_utils import load_json, flatten
from ..utils.text_cleaner import TextCleaner
from ..utils.torch_utils import load_tokenizer

class Utterance():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def __repr__(self):
        return f'{self.speaker}: {self.text}'
    
class Conversation():
    def __init__(self, data:dict):
        for k, v in data.items():
            if k == 'utterances':
                self.utts = [Utterance(**utt) for utt in v]
            else:
                setattr(self, k, v)
                
    def __iter__(self):
        return iter(self.utts)

    def __getitem__(self, k):
        return self.utts[k]

    def __len__(self):
        return len(self.utts)

class ConvHelper:    
    def __init__(self, trans_name:str, filters:list=None, tqdm_disable:bool=False):
        """ Initialises the Conversation helper """
        
        self.trans_name = trans_name

        if trans_name:
            self.tokenizer = load_tokenizer(trans_name)
                                                
        self.cleaner = TextCleaner(filters=filters)
        self.tqdm_disable = tqdm_disable
    
    def prepare_data(self, path:str, lim:int=None, quiet=False)->List:
        """ Given path, will load json and process data for downstream tasks """
        
        assert path.split('.')[-1] == 'json', "data must be in json format"

        raw_data = load_json(path)
        self.labels = self.load_label_info(path)

        data = [Conversation(conv) for conv in raw_data]
        data = self.process_data(data, lim, quiet)
        return data
    
    def process_data(self, data, lim=None, quiet=False):
        if lim: data = data[:lim]
        self.clean_text(data)
        self.tok_convs(data, quiet)  
        return data
    
    @staticmethod
    def load_label_info(path:str)->dict:
        #replace filename before extension with `labels'
        label_path = re.sub(r'\/(\w*?)\.json', '/labels.json', path)
        if os.path.isfile(label_path): 
            label_dict = load_json(label_path)
            label_dict = {int(k):v for k, v in label_dict.items()}
            return label_dict
               
    def clean_text(self, data:List[Conversation]):
        """ processes text depending on arguments. E.g. punct=True filters punctuation"""
        for conv in data:
            for utt in conv:
                utt.text = self.cleaner.clean_text(utt.text)
    
    def tok_convs(self, data:List[Conversation], quiet=False):
        """ generates tokenized ids for each utterance in Conversation """
        for conv in tqdm(data, disable=quiet):
            for utt in conv.utts:
                utt.ids = self.tokenizer(f'{utt.speaker}: {utt.text}').input_ids
            
    def __getitem__(self, x:str):
        """ returns conv with a given conv_id if exists in dataset """
        for conv in self.data:
            if conv.conv_id == str(x): return conv
        raise ValueError('conversation not found')
             
    def __contains__(self, x:str):
        """ checks if conv_id exists in dataset """
        output = False
        if x in [conv.conv_id for conv in self.data]:
            output = True
        return output
