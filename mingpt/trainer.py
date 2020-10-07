"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F

logger = logging.getLogger(__name__)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader
    
    block_size = 4

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, models, decmodel, modelcount, internalModelCount, train_dataset, test_dataset, config):
       
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        # Default: CPU
        self.device = 'cpu'
        self.modelcount = modelcount
        self.internalModelCount = internalModelCount
        self.models = []
        self.decmodel = decmodel
        self.bMultipleOuts = config.bMultipleOuts
        
        # take over whatever gpus are on the system
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.decmodel = torch.nn.DataParallel(self.decmodel).to(self.device)
        
        for m in range(self.modelcount):
            self.models.append(models[m])
            # take over whatever gpus are on the system
            if torch.cuda.is_available():
                self.models[m] = torch.nn.DataParallel(self.models[m]).to(self.device)

    def save_checkpoint(self):
        if self.config.ckpt_path is not None:
            # DataParallel wrappers keep raw model object in .module attribute
            ckpt_models = []
            for m in range(self.modelcount):
                ckpt_models.append(self.models[m].module if hasattr(self.models[m], "module") else self.models[m])
                logger.info("saving %s", self.config.ckpt_path)
                torch.save(ckpt_models[m].state_dict(), self.config.ckpt_path)

    def train(self):
        config = self.config
        models = []
        optimizers = []
        trainedmodel = []
        bias = []
        modelx = []
        modely = []
        
        no_decay = ["bias", "LayerNorm.weight", "ln1", "ln2", "ln_f", "rezero"]
        decmodel = self.decmodel
        
        optimizerdec = decmodel.configure_optimizers(config)
        for m in range(self.modelcount):
            trainedmodel.append(0)
            models.append(self.models[m])
            optimizers.append(models[m].configure_optimizers(config))
            bias.append(1000.0)

        def run_epoch(split):
            is_train = split == 'train'
            losses = []
            bestModel = 0
            
            decmodel.train(is_train)   
            for m in range(self.modelcount):
                models[m].train(is_train)
                losses.append([])
                
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=False, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)
           
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            
            for it, (x, y, z) in pbar:

                bestloss = 0
                
                # place data on the correct device
                x_cond = x if x.size(1) <= config.block_size else x[:, -config.block_size:] # crop context if needed
                y_cond = y if y.size(1) <= config.block_size else y[:, -config.block_size:] # crop context if needed
        
                x_cond = x_cond.to(self.device)
                y_cond = y_cond.to(self.device)
               
                # forward the decision model
                if self.modelcount > 1:
                    with torch.set_grad_enabled(is_train):

                        # Sample and train
                        dlogits, dloss = decmodel(x_cond, y_cond) # Target here is not next word, but next word after that
                        
                        # Sample
                        samplelogits = dlogits[:, -1, :]
                        sampleprobs = F.softmax(samplelogits, dim=-1)
                        modelID = torch.multinomial(sampleprobs, num_samples=1)
                        modelID = modelID[0].item()
                        bestModel = modelID % self.modelcount

                if is_train:
                    
                    # Train
                    if self.modelcount > 1:
                        dloss = dloss.mean()
                        decmodel.zero_grad()
                        dloss.backward()
                        torch.nn.utils.clip_grad_norm_(decmodel.parameters(), config.grad_norm_clip)
                        optimizerdec.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for m in range(self.modelcount):
                            for param_group in optimizers[m].param_groups:
                                param_group['lr'] = lr
                                
                        for param_group in optimizerdec.param_groups:
                                param_group['lr'] = lr
                                
                    else:
                        lr = config.learning_rate
  
                     # backprop and update the parameters
    
                    for m in range(self.modelcount):
                        if m == bestModel:
                            
                            # place data on the correct device
                            x_cond = x if x.size(1) <= config.block_size else x[:, -config.block_size:] # crop context if needed
                            y_cond = y if y.size(1) <= config.block_size else y[:, -config.block_size:] # crop context if needed

                            x_cond = x_cond.to(self.device)
                            y_cond = y_cond.to(self.device)

                            # forward the decision model
                            with torch.set_grad_enabled(is_train):
                                
                                if self.bMultipleOuts == True:
                                    y_cond = z
                                    y_cond[0] = y_cond[0].to(self.device)
                                    y_cond[1] = y_cond[1].to(self.device)
                                    y_cond[2] = y_cond[2].to(self.device)
                                    y_cond[3] = y_cond[3].to(self.device)
                                    y_cond[4] = y_cond[4].to(self.device)
                                    
                                rlogits, rloss = models[m](x_cond, y_cond)
                                rloss = rloss.mean() # collapse all losses if they are scattered on multiple gpus
                                losses.append(rloss.item())
                                
                            models[m].zero_grad()
                            rloss.backward()
                            torch.nn.utils.clip_grad_norm_(models[m].parameters(), config.grad_norm_clip)
                            optimizers[m].step()
                        
                            trainedmodel[m] = trainedmodel[m] + 1
                            bias[m] = bias[m] * 1.01
                            
                        else:
                            bias[m] = bias[m] * (1.0 - 0.01 / self.modelcount)

                    # report progress
                    if self.modelcount > 1:
                        pbar.set_description(f"epoch {epoch+1} iter {it}: tl {rloss.item():.4f}. brtl {dloss.item():.4f}. bestm: {bestModel}. bim: {modelID}. lr {lr:e}")
                    else:
                        pbar.set_description(f"epoch {epoch+1} iter {it}: tl {rloss.item():.4f}. lr {lr:e}")

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss

        best_loss = float('inf')
        self.tokens = 0 # counter used for learning rate decay
        for epoch in range(config.max_epochs):

            run_epoch('train')
            if self.test_dataset is not None:
                test_loss = run_epoch('test')

            for m in range(self.modelcount):
                print(f"Model: {m}, trained iter: {trainedmodel[m]}" )
                
            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss
                self.save_checkpoint()
