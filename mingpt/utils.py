import random
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

def defragTensorMulInst(tensorchunk):
    counters = {}
    imul = []
    tccounter = 0
    for s in tensorchunk[0]:
        idx = s.item()
        idx = 5 * math.floor(idx/5) # decrease to nearest token
        if idx in counters:
            counters[idx] = min(5, counters[idx] + 1)
        else:
            counters[idx] = 1

        tensorchunk[0][tccounter] = idx + counters[idx] - 1
        tccounter += 1
        
    return tensorchunk

@torch.no_grad()
def sampleMultiModelProb(models, decmodel, modelcount, internalModelCount, x, steps, temperature=1.0, sample=False, top_k=None, bMultipleIns=False, bMultipleOuts=False):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = []
    sampledmodel = []
    bias = []
    modelx = []
    
    decmodel.eval()
    for m in range(modelcount):
        sampledmodel.append(0)
        bias.append(1000.0)
        block_size.append(models[m].get_block_size())
        models[m].eval()
    
    for k in range(steps):
        probs = []
        bestv = 0
        bestm = 0
        bestprob = 0
        
        # Sample from Decisionmodel
        x_cond = x if x.size(1) <= block_size[m] else x[:, -block_size[m]:] # crop context if needed
        
        # If multiple instances per token -> put them in order
        if bMultipleIns:
            x_cond = defragTensorMulInst(x_cond)
        dlogits, _ = decmodel(x_cond) # Target here is not next word, but next word after that

        # Sample
        samplelogits = dlogits[:, -1, :]
        sampleprobs = F.softmax(samplelogits, dim=-1)
        modelID = torch.multinomial(sampleprobs, num_samples=1)
        modelID = modelID[0].item()
        modelID = modelID % modelcount
        
        m = modelID
        x_cond = x if x.size(1) <= block_size[m] else x[:, -block_size[m]:] # crop context if needed

        # If multiple instances per token -> put them in order
        if bMultipleIns:
            x_cond = defragTensorMulInst(x_cond)
        
        logits, _ = models[m](x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature

        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
   
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
            
        if bMultipleOuts:
            ix = torch.tensor([[5 * math.floor(ix.item()/5)]], dtype=torch.long) # decrease to nearest token
            
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)
        sampledmodel[m] = sampledmodel[m]+1

    for m in range(modelcount):
        print(f"Model: {m}, sampled iter: {sampledmodel[m]}, train bias: {bias[m]}" )
    return x
