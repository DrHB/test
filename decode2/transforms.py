__all__ = ['print_clas_signature', 'TransformBase', 'ScaleTensor', 'RandomCrop3D']

# Cell
import inspect
import torch
from scipy.ndimage import gaussian_filter
from .utils import *
import random

# Cell
def print_clas_signature(self, nms):
    "print class signature"
    mod = inspect.currentframe().f_back.f_locals
    for n in nms:
        print(f'{n}: {getattr(self,n)}')

class TransformBase:
    '''
    All transformations optianally must be inherited from this class for ncie
    representations and checks if input to given transformations is tensor

    '''
    def __repr__(self):
        print (f'Transform({self.__class__.__name__})')
        name = inspect.signature(self.__class__).parameters.keys()
        print_clas_signature(self, name)
        return ''

    def __call__(self, x):
        tst_check_tensor(x)

    @staticmethod
    def _get_device(x):
        return getattr(x, 'device')



# Cell
class ScaleTensor(TransformBase):
    """
    \nScales given `torch.Tensor` between `low` and `high`

    \nParameters:
    \n`low`     : lower bound
    \n`high`    : upper bound
    \n`data_min`: max value of data
    \n`data_max`: min value of main data

    \nReturns:
    \nScaled tensor

    """
    def __init__(self, low: float, high: float, data_min: float=0., data_max: float =1.):
        self.low = low
        self.high = high
        self.data_min = data_min
        self.data_max = data_max
        self.ratio = (self.high-self.low) /(self.data_max-self.data_min)

    def __call__(self, x) -> torch.Tensor:
        super().__call__(x)
        return self.ratio * x + self.low- self.data_min * self.ratio
    

class EstimateBackground(TransformBase):
    def __init__(self, smoothing_filter_size, div_factor=1):
        self.smoothing_filter_size = smoothing_filter_size
        self.div_factor = div_factor
    
    def __call__(self, image):
        background = gaussian_filter(image, self.smoothing_filter_size)/self.div_factor
        #background.clamp_min_(1.)
        return torch.tensor(background)
    
    
class RandomRoI(TransformBase):
    def __init__(self, roi_list: list):
        self.roi_list = roi_list
        
    def __call__(self, image):
        roi = random.choice(self.roi_list)
        x_l, x_r, y_l, y_r = roi
        return image[:, :, x_l:x_r, y_l:y_r]
        


# Cell
class RandomCrop3D(TransformBase):
    """
    Ramdomly Crops 3D tensor.

    \nThis class will generate random crop of `crop_sz`. This calss is initilize
    with `img_sz` which should be a demension of 4 [Channel, Height, Width, Depth] and
    a `crop_sz` dimesnion of 3 [Height, Width, Depth] of desired crop. For each crop
    dimension `_get_slice` function will calculate random int ranging from 0 to (img_sz-crop_sz).
    and return tuple of containing two slice intergers. If one dimension of `img_sz` matches
    one dimension of `crop_sz` the resulting tuple will be `(None, None)` which will result
    in not croping this particular dimension.


    \nParameters:
    \n`img_sz`     : Size of the 3D image `(C, H, W, D)`
    \n`crop_sz`    : Size of the 3D crop  `(H, W, D)`

    \nReturns:
    \nCroped 3D image of the given `crop_sz`

    """
    def __init__(self, img_sz, crop_sz):
        assert len(img_sz)  == 4 , f'Lenth of img_sz  should be 4 - (C, H, W, D) and not {len(img_sz)}'
        assert len(crop_sz) == 3 , f'Lenth of crop_sz should be 3 not {len(crop_sz)}'
        _, h, w, d = img_sz
        self.img_sz  = tuple((h, w, d))
        self.crop_sz = tuple(crop_sz)
        assert (self.img_sz) >  self.crop_sz


    def __call__(self, x):
        super().__call__(x)
        slice_hwd = [self._get_slice(i, k) for i, k in zip(self.img_sz, self.crop_sz)]
        return self._crop(x, *slice_hwd)


    @staticmethod
    def _get_slice(sz, crop_sz):
        up_bound = sz-crop_sz
        if  up_bound == 0:
            return None, None
        else:
            l_bound = torch.randint(up_bound, (1,))
        return l_bound, l_bound + crop_sz

    @staticmethod
    def _crop(x, slice_h, slice_w, slice_d):
        return x[:, slice_h[0]:slice_h[1], slice_w[0]:slice_w[1], slice_d[0]:slice_d[1]]