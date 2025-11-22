# credits: https://github.com/yulun-rayn/CS294-158/blob/main/deepul/models/histogram.py
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "Histogram",
]

class Histogram(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(d), requires_grad=True)
    
    def forward(self, x: torch.Tensor):
        logits = self.logits.expand(x.size(0), -1)
        return logits
    
    def loss(self, x: torch.Tensor):
        x = x.to(next(self.parameters()).device)
        return F.cross_entropy(self(x), x)

    def get_probs(self, tau=1.0):
        return F.softmax(self.logits / tau, dim=0).detach().cpu().numpy()