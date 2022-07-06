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
    def __init__(self, trans_name:str, filters:list=None, 
                 num_speakers:int=None, tqdm_disable:bool=False):
        """ Initialises the Conversation helper """
        
        self.trans_name = trans_name
        self.special_tok = False #whether speaker tokens are added

        if trans_name:
            self.tokenizer = load_tokenizer(trans_name)
            if num_speakers: self.register_speakers(num_speakers)
                                                
        self.cleaner = TextCleaner(filters=filters)
        self.tqdm_disable = tqdm_disable
    
    def register_speakers(self, num_speakers:int=2):
        self.speaker_tokens = []
        
        for i in range(num_speakers):
            spkr_text = f'[SPKR_{i}]'
            self.tokenizer.add_tokens(spkr_text, special_tokens=True)
            spkr_token = self.tokenizer(spkr_text, add_special_tokens=True).input_ids[0]
            self.speaker_tokens.append(spkr_token)
        
        print(f'added {len(self.speaker_tokens)} speaker tokens')
        
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
        self.get_speaker_ids(data)
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
        """ processes text depending on arguments. E.g. punct=True filters
        punctuation, action=True filters actions etc."""
        for conv in data:
            for utt in conv:
                utt.text = self.cleaner.clean_text(utt.text)
    
    def tok_convs(self, data:List[Conversation], quiet=False):
        """ generates tokenized ids for each utterance in Conversation """
        for conv in tqdm(data, disable=quiet):
            for utt in conv.utts:
                utt.ids = self.tokenizer(utt.text).input_ids
            
    def get_speaker_ids(self, data:List[Conversation]):
        speakers = Counter([utt.speaker for conv in data for utt in conv])
        speakers, _ = zip(*speakers.most_common())
        self.speaker_dict = {s:k for k, s in enumerate(speakers)}

        for conv in data:
            for utt in conv:
                spkr_id = self.speaker_dict[utt.speaker]
                utt.spkr_id = spkr_id
    
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
