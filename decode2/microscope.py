__all__ = ['Microscope', 'place_psf', '_place_psf', 'extractvalues']

import torch
import torch.nn.functional as F
from torch import nn
from torch.jit import script
from typing import Union, List


class Microscope(nn.Module):
    """
    Mircoscope module takes  4 volumes 'locs_3d', 'x_os_3d', 'y_os_3d', 'z_os_3d',
    'ints_3d'  and applies point spread function:
    1) Extracts values of intensities and offsets of given emitters
    2) Applies parametric PSF if given
    3) Applies empirical  PSF if given
    4) Combine all PSF and multiply by intenseties
    5) Normalize PSFs
    6) Places point spread function on sampled locations 'locs_3d' to
    generate 'x_sim' simulated image


    Args:
        parametric_psf (torch.nn.Module): List of Paramateric PSF
        empirical_psf (torch.nn.Module): List of Emperical PSF
        noise (torch.nn.Module): Camera noise model
        scale(float): Paramter for scaling point spread functions

    Shape:
        -Input: locs_3d: (BS, C, H, W, D)
                x_os_3d: (BS, C, H, W, D)
                y_os_3d: (BS, C, H, W, D)
                z_os_3d: (BS, C, H, W, D)
                ints_3d: (BS, C, H, W, D)

        -Output: xsim:    (BS, C, H, W, D)
    """


    def __init__(self,
                 parametric_psf: List[torch.nn.Module]=None,
                 empirical_psf : List[torch.nn.Module] = None ,
                 noise: Union[torch.nn.Module,  None]=None,
                 scale: float = 10.,
                 ):

        super().__init__()
        self.parametric_psf = parametric_psf if parametric_psf else None
        self.empirical_psf = empirical_psf if empirical_psf else None
        self.scale = torch.nn.Parameter(torch.tensor(scale))
        self.noise = noise

    def forward(self, locs_3d, x_os_3d, y_os_3d, z_os_3d, ints_3d, bg=None,):

        x_os_val, y_os_val, z_os_val, i_val = extractvalues(locs_3d,
                                                            x_os_3d,
                                                            y_os_3d,
                                                            z_os_3d,
                                                            ints_3d)
        #if there is no emitters or lacation we just return xsim (we dont have to apply psf)
        if x_os_val.shape[0]==0:
            xsim = locs_3d
            return xsim
        
        psf = 0
        if self.parametric_psf:
            for param_psf_ in self.parametric_psf:
                psf += param_psf_(x_os_val, y_os_val, z_os_val)

        if self.empirical_psf:
            for emper_psf in self.empirical_psf:
                psf += emper_psf(x_os_val, y_os_val, z_os_val)

        #normalizing psf
        psf = psf.div(psf.sum(dim=[1, 2, 3], keepdim=True))
        #applying intenseties
        psf = psf * i_val.reshape(-1, 1, 1, 1)
        xsim = place_psf(locs_3d, psf)
        xsim = 1000*self.scale * xsim
        torch.clamp_min_(xsim,0)
       
        return xsim

def place_psf(locs_3d, psf_volume):
    """
    Places point spread function to corresponding location in 3d volume in
    follwing 3 steps.

    1) Paddes locs_3d with padding to ensure that there is
    enough space on the edge when copying PSF.
    2) Extracts coordinates
    where emitters are present in padded 'pd_locs_3d' and assign
    there coordinates to `b`, `c`, `h`, `w` and `d`.
    3) This things get passed to jit optimized _place_psf whcih loops thru
    original locs_3d and places psfs.

    Args:
        locs_3d: torch.Tensor
        psf_volume: torch.Tensor

    Shape:
        -Input: locs_3d: (BS, C, H, W, D)
                psf: (Num_E, PSF_SZ_X, PSF_SZ_Y, PSF_SZ_Z) [
                Num_E-Number of Emitters, PSF_SZ_{X, Y, Z} - PSF filter size]
        -Output: locs_3d_psf:    (BS, C, H, W, D)

    Returns:
        locs_3d_psf
    """

    filter_size = psf_volume.shape[1:]
    filter_sizes = torch.cat(
        [torch.tensor((sz // 2, sz // 2 + 1)) for sz in filter_size]).reshape(
        3, 2)
    padding_sz = torch.tensor(max(filter_size) // 2 + 2)
    pd_locs3d = F.pad(input=locs_3d, pad=tuple([padding_sz] * 6), value=0.)
    b, c, h, w, d = tuple(pd_locs3d.nonzero().transpose(1, 0))
    locs_3d_psf = _place_psf(pd_locs3d, psf_volume, padding_sz,
                             filter_sizes, b, c, h, w, d)
    return locs_3d_psf


@script
def _place_psf(pd_locs3d, volume, pad_size, fz, b, c, h, w, d):
    # create empty tensor
    loc3d_like = torch.zeros_like(pd_locs3d)

    # place psf
    for idx in range(b.shape[0]):
        loc3d_like[b[idx], c[idx],
        h[idx] - fz[0][0]:h[idx] + fz[0][1],
        w[idx] - fz[1][0]:w[idx] + fz[1][1],
        d[idx] - fz[2][0]:d[idx] + fz[2][1]] += volume[idx]

    b_sz, ch_sz, h_sz, w_sz, d_sz = loc3d_like.shape

    # unpad to original size
    loc3d_like_unpad = loc3d_like[:, :, pad_size: h_sz - pad_size,
                                  pad_size: w_sz - pad_size,
                                  pad_size: d_sz - pad_size]
    return loc3d_like_unpad

def extractvalues( locs: torch.tensor,
                    x_os: torch.tensor,
                    y_os: torch.tensor,
                    z_os: torch.tensor,
                    ints:torch.tensor, dim: int=3):
    """
    Extracts Values of intensities and offsets of given emitters

    This function will take `locs`, `x_os`, `y_os`, `z_os`, `ints` all of
    shape and will  extract `coord` cordinate of locations where our
    emittors  are present. This `coord` will be used to extract values of
    `x`, `y`, `z`, offsets and intensities - `i` where emitter is present.

    Args:
        locs: location
        x_os: X offset
        y_os: Y offset
        z_os: Z offset
        ints: Intenseties
        dim:  Dimension 2D or 3D

    Shape:
        -Input: locs_3d: (BS, C, H, W, D)
                x_os_3d: (BS, C, H, W, D)
                y_os_3d: (BS, C, H, W, D)
                z_os_3d: (BS, C, H, W, D)
                ints_3d: (BS, C, H, W, D)

        -Output: :
                x_os_val: (Num_E, 1, 1, 1)
                y_os_val: (Num_E, 1, 1, 1)
                z_os_val: (Num_E, 1, 1, 1)
                ints_val: (Num_E, 1, 1, 1)


    """

    dim = tuple([1 for i in range(dim)])
    coord = tuple(locs.nonzero().transpose(1,0))
    x_os_val = x_os[coord].reshape(-1, *dim)
    y_os_val = y_os[coord].reshape(-1, *dim)
    z_os_val = z_os[coord].reshape(-1, *dim)
    ints_val = ints[coord].reshape(-1, *dim)
    return  x_os_val, y_os_val, z_os_val, ints_val
