__all__ = ['img_to_coord', 'get_true_labels', 'tiff_imread', 'hasattrs', 'show_image', 'AverageMeter', 'tst_check_tensor']

import torch
from tifffile import imread
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import numpy as np

def count_nonzero(x):
    'counts nonzero in tensor'
    return x[x.nonzero(as_tuple=True)].shape


def jaccard_coeff(logits, target, thresh=0.2):
    "pytorch implemenattion of jaccard"
    input = (torch.sigmoid(logits)>thresh).float()
    eps = 1e-15
    input = input.view(-1)
    target = target.view(-1)
    intersection = (input * target).sum()
    union = (input.sum() + target.sum()) - intersection
    return (intersection / (union + eps))


def f1_scores(logits, target, thresh=0.2):
    "pytorch implemenattion of jaccard"
    y_pred = (torch.sigmoid(logits)>thresh).float().view(-1)
    target = target.view(-1)
    tp = (target * y_pred).sum().to(torch.float32)
    tn = ((1 - target) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - target) * y_pred).sum().to(torch.float32)
    fn = (target * (1 - y_pred)).sum().to(torch.float32) 
    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    return f1.item(), precision.item(), recall.item()


def img_to_coord(locs, x_os, y_os, z_os, *args):
    """
    Given `locs` or probabilites will extract value of x_os, y_os, z_os where probability is more than 0.
    also generates counts of location and returns mask 
    """

    s_inds    = tuple(locs.nonzero().transpose(1,0))
    
    x =  x_os[s_inds] + s_inds[2].type(torch.cuda.FloatTensor) + 0.5 
    y =  y_os[s_inds] + s_inds[3].type(torch.cuda.FloatTensor) + 0.5 
    z =  z_os[s_inds] + s_inds[4].type(torch.cuda.FloatTensor) + 0.5 
    a = [item[s_inds].unsqueeze(-1) for item in args]
    xyz =  torch.stack((x, y, z), dim=1)
    
    #to match where in batch we have a counts (if there is no count at this 
    #position we will get 0 in tensor
    counts_ = torch.unique_consecutive(s_inds[0], return_counts=True)[1]
    bsz_loc = torch.unique(s_inds[0])
    bs = locs.shape[0]
    counts = torch.cuda.LongTensor(bs).fill_(0)
    counts[bsz_loc] = counts_
    
    max_counts    = counts.max()
    if max_counts==0: max_counts = 1 #if all 0 will return empty matrix of correct size
    xyz_list = torch.cuda.FloatTensor(x_os.shape[0],max_counts,3).fill_(0)
    i_list   = [torch.cuda.FloatTensor(x_os.shape[0],max_counts,1).fill_(0) for i in range(len(a))]
    s_arr    = torch.cat([torch.arange(c) for c in counts], dim = 0)
    s_mask   = torch.cuda.FloatTensor(locs.shape[0],max_counts).fill_(0)
    s_mask[s_inds[0],s_arr] = 1
    xyz_list[s_inds[0],s_arr] = xyz
    for i,k in zip(i_list, a): i[s_inds[0],s_arr] = k
    return (xyz_list, counts, s_mask) + tuple(i_list)


def get_true_labels(locs, x_os, y_os, z_os, ints):
    xyz_list, counts_true, s_mask, i_1 = img_to_coord(locs, x_os, y_os, z_os, ints)
    xyzi_true = torch.cat((xyz_list, i_1), dim=-1)
    return xyzi_true, counts_true, s_mask


# Cell
def tiff_imread(path):
    '''helper function to read tiff file with pathlib object or str'''
    if isinstance(path, str) : return imread(path)
    if isinstance(path, Path): return imread(str(path))


def hasattrs(o,attrs):
    "checks of o has several attrs"
    return all(hasattr(o,attr) for attr in attrs)


def show_image(im, ax=None, title=None, figsize=(4, 5), **kwargs,):
    'plots image from nump or tensor'
    if hasattrs(im, ('data','cpu','permute')):
        im = im.data.cpu()
        if im.shape[0]<5: im=im.permute(1,2,0)
    elif not isinstance(im,np.ndarray): im=array(im)
    if im.shape[-1]==1: im=im[...,0]
    if ax is None: _,ax = plt.subplots(figsize=figsize)
    ax.imshow(im, **kwargs)
    if title is not None: ax.set_title(title)
    ax.axis('off')
    return ax


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Cell
def tst_check_tensor(x):
    "cehcks if x is torch.Tensor"
    assert isinstance(x, torch.Tensor), f'must be torch.tensor not {type(x)}'
    return


def one_batch(dl):
    "returns one batch"
    return next(iter(dl))