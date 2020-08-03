__all__ = ['PointProcessUniform', 'PointProcessGaussian']

import torch
from torch import distributions as D, Tensor
from torch.distributions import Distribution
from .utils import img_to_coord, get_true_labels


class PointProcessUniform(Distribution):
    """
    This class is part of generative model and uses probility local_rate to
    generate locations `locations`  `x`, `y`, `z` offsets and `intensities` intensity of
    emitters. local_rate  should be `torch.tensor` scaled from 0.001 to 1),
    which is used by `_sample_bin` to generate `0` and `1` . `0` means that
    we dont have emitter at given pixel and 1 means emitters is presnet.This
    map is used  to generate offset in `x`, `y`, `z` and intensities  which
    tells how bright is emitter or in some casses how many emitters are
    bound to RNA molecules.

    Args:
        local_rate (BS, C, H, W, D): Local rate
        min_int (int): minimum intensity of emitters
        bg(bool): if returns sampled backround

    """


    def __init__(self, local_rate: torch.tensor, min_int: float):

        self.local_rate = local_rate
        self.device = self._get_device(self.local_rate)
        self.min_int = torch.tensor(min_int, device=self.device)

    def sample(self, N:int =1):

        if N == 1: return self._sample()
        else:
            res_ = [self._sample() for i in range(N)]
            locations = torch.cat([i[0] for i in res_], dim=1)
            x_offset = torch.cat([i[1] for i in res_], dim=1)
            y_offset = torch.cat([i[2] for i in res_], dim=1)
            z_offset = torch.cat([i[3] for i in res_], dim=1)
            intensities = torch.cat([i[4] for i in res_], dim=1)
            return locations, x_offset, y_offset, z_offset, intensities

    def _sample(self):

        sample_shape = self.local_rate.shape
        locations = D.Bernoulli(self.local_rate).sample()
        zero_point_five = torch.tensor(0.5, device=self.device)
        x_offset = D.Uniform(low=0 - zero_point_five,
                         high=0 + zero_point_five, ).sample(
            sample_shape=sample_shape)
        y_offset = D.Uniform(low=0 - zero_point_five,
                         high=0 + zero_point_five, ).sample(
            sample_shape=sample_shape)
        z_offset = D.Uniform(low=0 - zero_point_five,
                         high=0 + zero_point_five, ).sample(
            sample_shape=sample_shape)
        intensities = D.Uniform(low=self.min_int, high=1.0).sample(
            sample_shape=sample_shape)
        x_offset *= locations
        y_offset *= locations
        z_offset *= locations
        intensities *= locations
        return locations, x_offset, y_offset, z_offset, intensities

    def log_prob(self, locations_3d, x_offset_3d=None, y_offset_3d=None, z_offset_3d=None, intensities_3d=None):
        log_prob = D.Bernoulli(self.local_rate).log_prob(locations_3d)
        return log_prob

    @staticmethod
    def _get_device(x):
        return getattr(x, 'device')


class PointProcessGaussian(Distribution):
    def __init__(self, logits: torch.tensor, xyzi_mu: torch.tensor,
                 xyzi_sigma: torch.tensor):
        "logits: BS, C, H, W, D"
        self.logits = logits
        self.xyzi_mu = xyzi_mu
        self.xyzi_sigma = xyzi_sigma

    def sample(self, N:int =1):
        if N ==1:
            return self._sample()
       
        res_ = [self._sample() for i in range(N)]
        locations = torch.cat([i[0] for i in res_], dim=1)
        x_offset = torch.cat([i[1] for i in res_], dim=1)
        y_offset = torch.cat([i[2] for i in res_], dim=1)
        z_offset = torch.cat([i[3] for i in res_], dim=1)
        intensities = torch.cat([i[4] for i in res_], dim=1)
        return locations, x_offset, y_offset, z_offset, intensities



    def _sample(self):
        locations = D.Bernoulli(logits=self.logits).sample()
        xyzi = D.Independent(D.Normal(self.xyzi_mu, self.xyzi_sigma),
                             1).sample()
        x_offset, y_offset, z_offset, intensities = (i.unsqueeze(1) for i in
                                  torch.unbind(xyzi, dim=1))
        #x_offset    *=locations
        #y_offset    *=locations
        #z_offset    *=locations
        #intensities *=locations
        
        return locations, x_offset, y_offset, z_offset, intensities

    def log_prob(self, locations_3d, x_offset_3d, y_offset_3d, z_offset_3d, intensities_3d):
        xyzi, counts, s_mask = get_true_labels(locations_3d, x_offset_3d, y_offset_3d, z_offset_3d, intensities_3d )
        x_mu, y_mu, z_mu, i_mu = (i.unsqueeze(1) for i in
                                  torch.unbind(self.xyzi_mu, dim=1))
        x_si, y_si, z_si, i_si = (i.unsqueeze(1) for i in
                                  torch.unbind(self.xyzi_sigma, dim=1))
        
        P = torch.sigmoid(self.logits) + 0.00001
        count_mean = P.sum(dim=[2, 3, 4]).squeeze(-1)
        count_var = (P - P ** 2).sum(dim=[2, 3, 4]).squeeze(-1)  #avoid situation where we have perfect match
        count_dist = D.Normal(count_mean, torch.sqrt(count_var))
        count_prob = count_dist.log_prob(counts)
        mixture_probs = P / P.sum(dim=[1, 2, 3], keepdim=True)
        
        xyz_mu_list, _, _, i_mu_list, x_sigma_list, y_sigma_list, z_sigma_list, i_sigma_list, mixture_probs_l = img_to_coord(
            P, x_mu, y_mu, z_mu, i_mu, x_si, y_si, z_si, i_si, mixture_probs)
        xyzi_mu = torch.cat((xyz_mu_list, i_mu_list), dim=-1)
        xyzi_sigma = torch.cat((x_sigma_list, y_sigma_list, z_sigma_list, i_sigma_list), dim=-1) #to avoind NAN
        mix = D.Categorical(mixture_probs_l.squeeze(-1))
        comp = D.Independent(D.Normal(xyzi_mu, xyzi_sigma), 1)
        spatial_gmm = D.MixtureSameFamily(mix, comp)
        spatial_prob = spatial_gmm.log_prob(xyzi.transpose(0, 1)).transpose(0,
                                                                            1)
        spatial_prob = (spatial_prob * s_mask).sum(-1)
        log_prob = count_prob + spatial_prob
        return log_prob
    

