import torch
import torch.nn as nn

from types import SimpleNamespace
from .utils.torch_utils import load_lm_transformer

class GPTLMModel(torch.nn.Module):
    """encodes all words in a conversation jointly in a transformer""" 
    def __init__(self, trans_name:str):
        super().__init__()
        loaded_weights = load_lm_transformer(trans_name)
        self.transformer = loaded_weights.transformer
        
        self.lm_head = loaded_weights.lm_head

    def forward(self, **kwargs):
        H = self.transformer(**kwargs)
        hidden_states = H.last_hidden_state
        logits = self.lm_head(hidden_states)
        return SimpleNamespace(H=H, logits=logits)