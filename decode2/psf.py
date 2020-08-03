__all__ = ['Gaus3D', 'get_grids', 'InterpolatedPSF']

import torch
from torch.jit import script
from torch import nn
import numpy as np


class Gaus3D(nn.Module):
    """
    Applies 3D gaussian point spread function
    This class is initiated by creating `x`, `y`, `z` filters. In
    forward pass from each x, y, z we substract corresponding `x_offset`,
    `y_offset`, `z_offset` offset values which is then divided by 3 learnable
    paramaters 'x', 'y', 'z' -scales. After this we apply psf and return
    torch.Tensor with the shape (Num_E, psf_size)

    Args:
        psf_size: Tuple[int, int, int]
        width: float
        x_scale: float
        y_scale: float
        z_scale: float

    Shape:
        -Input: x_offset_val: (Num_E, 1, 1, 1)
                y_offset_val: (Num_E, 1, 1, 1)
                z_offset_val: (Num_E, 1, 1, 1)

        -Output: psf_volume:    Num_E, PSF_SZ_X, PSF_SZ_Y, PSF_SZ_Z)

    """

    def __init__(self, psf_size: tuple, width: float, x_scale: float,
                 y_scale: float, z_scale: float):

        super().__init__()
        self.width = torch.nn.Parameter(torch.tensor(width))
        self.x_scale = torch.nn.Parameter(torch.tensor(x_scale))
        self.y_scale = torch.nn.Parameter(torch.tensor(y_scale))
        self.z_scale = torch.nn.Parameter(torch.tensor(z_scale))
        v = [torch.arange(0 - sz // 2, sz // 2 + 1) for sz in psf_size]
        self.register_buffer('x', v[0].reshape(1, 1, -1))
        self.register_buffer('y', v[1].reshape(1, -1, 1))
        self.register_buffer('z', v[2].reshape(-1, 1, 1))
        self.register_buffer('taylor_corr', torch.tensor(1 / 12))

    def forward(self, x_offset_val, y_offset_val, z_offset_val):
        x = (self.x - x_offset_val) ** 2
        y = (self.y - y_offset_val) ** 2
        z = (self.z - z_offset_val) ** 2
        w_x = x.div(self.x_scale)
        w_y = y.div(self.y_scale)
        w_z = z.div(self.z_scale)
        psf_volume = torch.exp(-(w_x + w_y + w_z) / (
                    2 * (w_z * self.width) ** 2 + self.taylor_corr))
        return psf_volume


class InterpolatedPSF(nn.Module):
    """

    \nInterpolates psf volume with offsets

    \nIn order to avoid `cuda` memory issues when stacking meshgrids, I wrote custom `jit` function `get_preds` which optimized stacking.

    \nParameters:
    \n`filter_size`: filter size of PSF

    \nReturns:
    3D interpolated psf volume with offests
    """

    def __init__(self, fs_x, fs_y, fs_z,upsample_factor=3, device='cuda'):
        super().__init__()
        self.upsampled_psf_size = list((upsample_factor*(np.array((fs_x, fs_y, fs_z))-1)+1).astype('int'))
        v = [torch.linspace(-1, 1, int(sz)) for sz in self.upsampled_psf_size]
        self.register_buffer('x', v[0])
        self.register_buffer('y', v[1])
        self.register_buffer('z', v[2])
        self.device=device
        self.psf_volume = nn.Parameter(torch.rand(1, *self.upsampled_psf_size))
        
    def forward(self, x_offset_val, y_offset_val, z_offset_val):
        N_emitters = x_offset_val.shape[0]
        x_offset = x_offset_val.view(-1) / self.upsampled_psf_size[0]
        y_offset = y_offset_val.view(-1) / self.upsampled_psf_size[1]
        z_offset = z_offset_val.view(-1) / self.upsampled_psf_size[2]
        x_offset = self.x.expand(x_offset.shape[0], self.upsampled_psf_size[0]).to(self.device) - x_offset.unsqueeze(1)
        y_offset = self.y.expand(y_offset.shape[0], self.upsampled_psf_size[1]).to(self.device) - y_offset.unsqueeze(1)
        z_offset = self.z.expand(z_offset.shape[0], self.upsampled_psf_size[2]).to(self.device) - z_offset.unsqueeze(1)

        x_offset, y_offset, z_offset = torch.meshgrid(x_offset.view(-1),
                                          y_offset.view(-1),
                                          z_offset.view(-1))
        
        m_grid = get_grids(x_offset, y_offset, z_offset, torch.tensor(self.upsampled_psf_size), torch.tensor(x_offset_val.shape[0]))
        
        
        grids = torch.nn.functional.grid_sample(self.psf_volume.repeat(N_emitters, 1, 1, 1, 1).to(self.device), m_grid)
        
        return grids.squeeze(1)
    

@script
def get_grids(z_offset, y_offset, x_offset, fs, E):
    empty = []
    for i in range(int(E)):
        ten_ = torch.stack((z_offset[i*fs[0]:(i+1)*fs[0], i*fs[1]:(i+1)*fs[1], i*fs[2]:(i+1)*fs[2]], 
                            y_offset[i*fs[0]:(i+1)*fs[0], i*fs[1]:(i+1)*fs[1], i*fs[2]:(i+1)*fs[2]], 
                            x_offset[i*fs[0]:(i+1)*fs[0], i*fs[1]:(i+1)*fs[1], i*fs[2]:(i+1)*fs[2]]),3)
        
        empty.append(ten_)
    return torch.stack(empty)