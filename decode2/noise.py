__all__ = ['sCMOS']

import torch
from torch import nn
from torch import distributions as D


class sCMOS(nn.Module):
    """
    Generates sCMOS noise distribution.
    Generates sCMOS noise distribution which can be used for sampling and
    calculating log probabilites.

    Args:
        theta (float): Paramter for gamma distribution
        background (float): background value
        baseline (float): basline

    Shape:
        -Input: x_sim: (BS, C, H, W, D)

        -Output: Gamma(concentration: (BS, C, H, W, D), rate: 
        (BS, C, H, W, D))
    """
    def __init__(self,
                 theta: float = 3.,
                 baseline: float = 0.01):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor(theta))
        self.register_buffer('baseline', torch.tensor(baseline))

    def forward(self, x_sim, background):
        x_sim_background = x_sim +  background
        x_sim_background.clamp_(1.0)
        conc = (x_sim_background - self.baseline) / self.theta
        xsim_dist = D.Gamma(concentration=conc,
                       rate=1 / self.theta)
        return xsim_dist