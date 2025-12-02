# Some credits: https://github.com/yulun-rayn/CS294-158/blob/main/deepul/models/mixture.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy

__all__ = [
    "DiscretizedLogisticMixture",
]
 
class DiscretizedLogisticMixture(nn.Module):
    def __init__(self, n_components: int, max_val: int) -> None:
        """
        Args:
            n_components (m): The number of components in the mix.
            max_val (d): The maximum discrete value of the data (e.g., 255 pixels, d classes).
        """
        super().__init__()
        self.m = n_components
        self.d = max_val

        # Logits (Unnormalized logarithm of the mixture coefficients)
        self.mixture_logits = nn.Parameter(torch.zeros(n_components))
        
        initials_means = torch.linspace(0, max_val, dtype=torch.float32, steps=n_components)
        self.means = nn.Parameter(initials_means)
        
        # Log Scales (Logarithmic scales, initialized to 0, i.e., standard deviation is 1)
        self.log_scales = nn.Parameter(torch.zeros(n_components))
        
    def _compute_log_probs(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.means.device).to(self.means.dtype)
        
        if x.dim() == 1:
            x = x.unsqueeze(-1)
            
        # x: [B, 1]
        # means: [1, m]
        means = self.means.view(1, -1)
        # log_scales: [1, m]
        log_scales = self.log_scales.view(1, -1)
        inv_std = torch.exp(-log_scales)
        
        # (x - mu) / sigma
        centered = x - means
        
        # plus: CDF(x + 0.5), min: CDF(x - 0.5)
        plus_in = inv_std * (centered + 0.5)
        min_in = inv_std * (centered - 0.5)
        
        # log(CDF(x)) = log_sigmoid(x)
        cdf_plus = torch.sigmoid(plus_in)
        cdf_min = torch.sigmoid(min_in)
        
        log_cdf_plus = plus_in - F.softplus(plus_in) # Equal to log_sigmoid
        log_one_minus_cdf_min = -F.softplus(min_in) # Equal to log(1-sigmoid)
        
        cdf_delta = cdf_plus - cdf_min
        log_probs_mid = torch.log(torch.clamp(cdf_delta, min=1e-12))
        
        log_probs = torch.where(
            x < 0.001, # Case: x == 0
            log_cdf_plus, # Log prob for x=0 edge
            torch.where(
                x > (self.d - 0.001), # Case: x == d
                log_one_minus_cdf_min, # Log prob for x=d edge
                log_probs_mid # Middle cases
            )
        )
        
        # Log P(x) = LogSumExp( log(weights) + log(P(x|component)) )
        mixture_log_probs = F.log_softmax(self.mixture_logits, dim=0)
        
        # [B, m] + [m] -> [B, m]
        total_log_terms = log_probs + mixture_log_probs.view(1, -1)
        
        # Ouput: log(sum(exp()))
        f_log_probs = torch.logsumexp(total_log_terms, dim=1)
        
        return f_log_probs
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(self._compute_log_probs(x))
    
    # NLL
    def loss(self, x: torch.Tensor) -> torch.Tensor:
        log_probs = self._compute_log_probs(x)
        return -torch.mean(log_probs)
    
    def get_probs(self) -> numpy.ndarray:
        x_all = torch.arange(self.d + 1, device=self.mixture_logits.device)
        return self.forward(x_all).detach().cpu().numpy()