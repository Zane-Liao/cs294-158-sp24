import torch
import torch.nn as nn
import torch.nn.functional as F

class Histogram(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(d), requires_grad=True)
    
    def forward(self, x: torch.Tensor):
        logits = self.logits.expand(x.size(0), -1)
        return logits
    
    def cross_loss(self, x: torch.Tensor):
        x = x.to(next(self.parameters()).device)
        return F.cross_entropy(self(x), x)

    def get_probs(self, tau=1.0):
        return F.softmax(self.logits / tau, dim=0).detach().cpu().numpy()