import torch
from typing import Callable

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import OpenAIGPTTokenizerFast, GPT2TokenizerFast 
from transformers import GPT2LMHeadModel, GPTNeoForCausalLM, OpenAIGPTLMHeadModel

def load_tokenizer(system:str)->'Tokenizer':
    """ downloads and returns the relevant pretrained tokenizer from huggingface """
    if   system == 'dialo_gpt_small' : tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    elif system == 'dialo_gpt_med'   : tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    elif system == 'dialo_gpt_large' : tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
    elif system == 'gpt2_small'      : tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    elif system == 'gpt2_med'        : tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-medium")
    elif system == 'gpt2_large'      : tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-large")
    elif system == 'gpt_small'       : tokenizer = OpenAIGPTTokenizerFast.from_pretrained("openai-gpt")
    elif system == 'gpt_neo'         : tokenizer = GPT2TokenizerFast.from_pretrained("EleutherAI/gpt-neo-1.3B")
    else: raise ValueError("invalid transfomer system provided")
    return tokenizer
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

def load_lm_transformer(system:str)->'Model':
    """ downloads and returns the relevant pretrained transformer from huggingface """
    if   system == 'dialo_gpt_small' : trans_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    elif system == 'dialo_gpt_med'   : trans_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    elif system == 'dialo_gpt_large' : trans_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
    elif system == 'gpt2_small'      : trans_model = GPT2LMHeadModel.from_pretrained("gpt2")
    elif system == 'gpt2_med'        : trans_model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
    elif system == 'gpt2_large'      : trans_model = GPT2LMHeadModel.from_pretrained("gpt2-large")
    elif system == 'gpt_small'       : trans_model = OpenAIGPTLMHeadModel.from_pretrained("openai-gpt")
    elif system == 'gpt_neo'         : trans_model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
    else: raise ValueError("invalid transfomer system provided")
    return trans_model

def no_grad(func:Callable)->Callable:
    """ decorator which detaches gradients """
    def inner(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return inner
