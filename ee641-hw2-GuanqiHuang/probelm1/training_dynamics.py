import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

@torch.no_grad()
def make_letter_classifier_embed(loader: DataLoader, device: torch.device):
    feats, labs = [], []
    for x, y, _ in loader:
        x = x.to(device)                             
        f = x.view(x.size(0), -1).float()           
        f = F.normalize(f, dim=1)
        feats.append(f)
        labs.append(y.to(device))
    feats = torch.cat(feats, 0)  
    labs = torch.cat(labs, 0)    
    return feats, labs

@torch.no_grad()
def assign_modes(gen_imgs: torch.Tensor, db_feats: torch.Tensor, db_labels: torch.Tensor):
    G = gen_imgs.view(gen_imgs.size(0), -1)
    G = F.normalize(G, dim=1)                
    sims = G @ db_feats.t()                  
    nn_idx = sims.argmax(dim=1)
    pred = db_labels[nn_idx]
    return pred

def log_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
