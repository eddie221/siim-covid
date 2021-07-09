#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 16:11:03 2021

@author: mmplab603
"""

from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T
from torch import nn, Tensor
from typing import List, Tuple, Dict, Optional
import torch

class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(self, image: Tensor,
                target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if target is not None:
                width, _ = F._get_image_size(image)
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
        return image, target
    
class Resize(object):
    def __init__(self, size):
        self.size = size
        
    def __call__(self, image, target):
        W, H = image.size
        image = F.resize(image, (self.size, self.size))
        
        bbox = target['boxes']
        bbox[:, 0] = bbox[:, 0] * self.size / W 
        bbox[:, 1] = bbox[:, 1] * self.size / H 
        bbox[:, 2] = bbox[:, 2] * self.size / W 
        bbox[:, 3] = bbox[:, 3] * self.size / H 
        target['boxes'] = bbox
        target['area'] = (bbox[:, 3] - bbox[:, 1]) * (bbox[:, 2] - bbox[:, 0])
        
        return image, target
    
class ToTensor(nn.Module):
    def forward(self, image: Tensor,
                target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.to_tensor(image)
        return image, target
    
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
    
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target