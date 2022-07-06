import wandb
import torch
import torch.nn.functional as F

from collections import namedtuple
from types import SimpleNamespace
from typing import List, Tuple

from .helpers import DirHelper, Batcher, ConvHelper
from .utils.torch_utils import no_grad
from .models import GPTLMModel

class Trainer():
    """"base class for running basic transformer classification models"""
    
    def __init__(self, exp_name:str, args:namedtuple):
        self.dir = DirHelper(exp_name)
        self.dir.save_args('model_args.json', args)
        self.set_up_helpers(args)
        
    ############  MAIN TRAIN LOOP  #################################
    
    def set_up_helpers(self, args:namedtuple):
        self.model_args = args
        
        self.C = ConvHelper(trans_name=args.transformer, 
                            num_speakers=args.num_speakers)

        self.batcher = Batcher(
                           C=self.C,
                           formatting=args.formatting,
                           max_len=args.max_len)
        
        self.model = GPTLMModel(trans_name=args.transformer, C=self.C)
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f'model has {num_params:,} parameters')

        self.device = args.device

    def train(self, args:namedtuple):
        self.dir.save_args('train_args.json', args)
        if args.wandb: self.set_up_wandb(args)
 
        train = self.C.prepare_data(args.train_path, args.lim)
        dev = self.C.prepare_data(args.dev_path, args.lim) if hasattr(args, 'dev_path') else None 

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)
        best_epoch = (-1, 10000, 0)
        self.to(self.device)
        
        for epoch in range(args.epochs):
            ######  TRAINING  ##############################
            self.model.train()
            self.dir.reset_metrics()
            train_b = self.batcher(data=train, bsz=args.bsz, shuffle=True)
            
            for k, batch in enumerate(train_b, start=1):
                output = self.model_output(batch)

                optimizer.zero_grad()
                output.loss.backward()
                optimizer.step()

                # accuracy logging
                self.dir.update_avg_metrics(loss=output.loss)

                # print train performance every now and then
                if k%args.print_len == 0:
                    perf = self.dir.print_perf('train', epoch, k)
                    if args.wandb:
                         wandb.log({'epoch':epoch, 'loss':perf.loss})
            
            ######  DEV  ##################################
            if dev:
                self.model.eval()
                perf = self.system_eval(dev, epoch)
                if args.wandb:
                    wandb.log({"dev_loss":perf.loss})

            # save performance if best dev performance 
            if perf.loss < best_epoch[1]:
                best_epoch = (epoch, perf.loss)
                if args.save: self.save_model()
      
            if epoch - best_epoch[0] >= 2:
                break
                
    def model_output(self, batch):
        if getattr(self, 'bias', False):
            return self.bias_model_output(batch)
             
        output = self.model(input_ids=batch.ids, 
                            attention_mask=batch.mask)
        
        loss = F.cross_entropy(output.logits.view(-1, output.logits.size(-1))
                               ,batch.labels.view(-1))
            
        # return accuracy metrics
        hits = torch.argmax(output.logits, dim=-1) == batch.labels
        hits = torch.sum(hits[batch.labels != -100]).item()
        num_preds = torch.sum(batch.labels != -100).item()

        return SimpleNamespace(loss=loss, logits=output.logits,
                               hits=hits, num_preds=num_preds)

    ############# EVAL METHODS ####################################
    @no_grad
    def system_eval(self, data, epoch:int):
        self.dir.reset_metrics()         
        batches = self.batcher(data=data, bsz=1, shuffle=False)
        for k, batch in enumerate(batches, start=1):
            output = self.model_output(batch)
            self.dir.update_avg_metrics(loss=output.loss)
            #removed acc line here
            
        perf = self.dir.print_perf('dev', epoch, 0)
        return perf    

    #############  MODEL UTILS  ###################################
    
    def save_model(self, name:str='base'):
        device = next(self.model.parameters()).device
        self.model.to("cpu")
        torch.save(self.model.state_dict(), 
                   f'{self.dir.abs_path}/models/{name}.pt')
        self.model.to(self.device)

    def load_model(self, name:str='base'):
        self.model.load_state_dict(
            torch.load(self.dir.abs_path + f'/models/{name}.pt'))

    def to(self, device):
        assert hasattr(self, 'model') and hasattr(self, 'batcher')
        self.model.to(device)
        self.batcher.to(device)

    ############  WANDB UTILS  ####################################
    
    def set_up_wandb(self, args:namedtuple):
        wandb.init(project=args.wandb, entity="adian", reinit=True,
                   name=self.dir.exp_name, dir=self.dir.abs_path)

        # save experiment config details
        cfg = {}
        
        cfg['formatting']  = self.model_args.formatting
        cfg['transformer'] = self.model_args.transformer
        cfg['epochs']      = args.epochs
        cfg['bsz']         = args.bsz
        cfg['lr']          = args.lr
        
        wandb.config.update(cfg) 
        wandb.watch(self.model)
        
    
