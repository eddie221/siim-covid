#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 16:12:25 2021

@author: mmplab603
"""

from transform import *
from dataloader import *
import sys
sys.setrecursionlimit(2000)

transform = Compose([Resize(416)])

dataset = siim_covid_Dataset("../../datasets/siim-covid19-detection/", phase = 'train', trans = transform)

print(dataset[0][1]['boxes'])