from pado.light import * 
from pado.optical_element import *
from pado.propagator import *
import torch
import numpy as np

def convert_resolution(param, args):
    # dataset
    if args.obstruction == 'fence':
        param.dataset_dir = '/projects/FHEIDE/obstruction_free_doe/Places365'
        param.data_resolution = [512,768]
    elif args.obstruction == 'raindrop' or 'dirt' or 'dirt_raindrop':
        param.training_dir = '/projects/FHEIDE/Bad2ClearWeather/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train'
        param.val_dir = '/projects/FHEIDE/Bad2ClearWeather/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val'
        param.data_resolution = [1024, 2048]
    else:
        assert False, "undefined obstruction"

    # convert resolution and pitch size
    param.equiv_image_size = param.img_res * param.image_sample_ratio # image resolution before downsampling in camera pixel pitch
    param.equiv_crop_size = int(param.equiv_image_size * param.camera_pitch / param.background_pitch)  # convert to background pixel pitch 
    return param

def randuni(low, high, size):
    '''uniformly sample from [low, high)'''
    return (torch.rand(size)*(high - low) + low)

def real2complex(real):
    return Complex(mag=real, ang=torch.zeros_like(real))

def compute_pad_size(current_size, target_size):
    assert current_size < target_size
    gap = target_size - current_size
    left = int(gap/2)
    right = gap - left
    return int(left), int(right)

def sample_psf(psf, sample_ratio):
    if sample_ratio == 1:
        return psf
    else:
        return torch.nn.AvgPool2d(sample_ratio, stride=sample_ratio)(psf)

def metric2pixel(metric, depth, args):
    return int(metric * args.param.focal_length / (depth * args.param.equiv_camera_pitch))

def edge_mask(R,cutoff, device):
    [x, y] = np.mgrid[-int(R):int(R),-int(R):int(R)]
    dist = np.sqrt(x**2 +y**2).astype(np.int32)
    mask = torch.tensor(1.0*(dist < cutoff))[None, None, ...]
    return mask.to(device)

class AttributeDict(dict):
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value