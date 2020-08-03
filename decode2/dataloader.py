__all__ = ['DecodeDataset']

# Cell

import torch


# Cell
class DecodeDataset:

    def __init__(self, tiff_stack: torch.Tensor, dataset_tfms: list, rate_transform, bg_transform, num_iter = 5000, device='cuda'):
        self.imgs = tiff_stack
        self.dataset_tfms = dataset_tfms
        self.num_iter = num_iter
        self.local_rate_tfms = rate_transform
        self.bg_transform = bg_transform
        self.device = 'cuda'

    def __len__(self):
        return self.num_iter

    def __getitem__(self, _):
        x = self._compose(self.imgs, self.dataset_tfms)
        local_rate = self.local_rate_tfms(x)
        background = self.bg_transform(x)
        
        return x.to(self.device), local_rate.to(self.device), background.to(self.device)

    def __repr__(self):
        print (f'{self.__class__.__name__} Summary:')
        print (f'tiff image stack: {self.imgs.shape}\n')
        print (f'Dataset tfms: {len(self.dataset_tfms)}')
        for i in self.dataset_tfms:
            print (f'\n-->')
            f"{i}"
        print (f'\nGenerative data tfms: {len(self.gen_tfms)}')
        for i in self.gen_tfms:
            print (f'\n-->')
            f"{i}"

        return ''

    @staticmethod
    def _compose(x, list_func):
        if not list_func: list_func.append(lambda x: x)
        for func in list_func:
            x = func(x)
        return x