#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 16:03:13 2021

@author: mmplab603
"""

import pydicom
import numpy as np
import glob
import matplotlib.pyplot as plt
from skimage.exposure import equalize_hist
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torch
import tqdm

# Load a DICOM file and get the pixels
def load_image(image_path):
    image = pydicom.dcmread(image_path)
    pixels = image.pixel_array

    min_pixel = np.min(pixels)
    max_pixel = np.max(pixels)

    if image.PhotometricInterpretation == "MONOCHROME1":
        pixels = max_pixel - pixels
    else:
        pixels = pixels
    return pixels

# Apply filters. Tweak params here.
def apply_filter(img):
    img = equalize_hist(img, nbins=256, mask=None)
    return img

class siim_covid_Dataset(Dataset):
    def __init__(self, path, phase, trans = None):
        self.image_paths = []# = glob.glob(os.path.join(path, "*/*/*.dcm"))
        self.study_label = pd.read_csv(os.path.join(path, "train_study_level.csv"))
        self.image_label = pd.read_csv(os.path.join(path, "train_image_level.csv"))
        self.study_label['StudyInstanceUID'] = self.study_label['id'].apply(lambda x : x[:-6])
        self.merge_data = self.image_label.merge(self.study_label, on='StudyInstanceUID')
        self.merge_data['ImageID'] = self.merge_data['id_x'].apply(lambda x : x[:-6])
        
        for i in tqdm.tqdm(range(len(self.merge_data))):
            instance_id = self.merge_data.iloc[i].StudyInstanceUID
            ImageID = self.merge_data.iloc[i].ImageID
            self.image_paths.append(glob.glob(os.path.join(path, phase, instance_id +"/*/{}.dcm".format(ImageID)))[0])
#         for instance_id in tqdm(self.merge_data):
#             images_path = glob.glob(os.path.join(path, instance_id +"/*/*"))
#             self.image_paths.append(glob.glob(os.path.join(path, instance_id +"/*/*")))
        
        self.merge_data['path'] = self.image_paths
        #self.merge_data.to_csv('/kaggle/working/study_image_merge.csv', index=False)
        self.merge_data = self.merge_data.drop(["id_x", "id_y"], axis = 1)
        
        self.trans = trans
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = load_image(self.merge_data.iloc[idx].path)
        image = apply_filter(image)
        image = Image.fromarray(image)
        bbox_data = self.merge_data.iloc[idx].label
        class_label = self.merge_data.iloc[idx]
        if int(class_label[3]):
            class_label = 0
        elif int(class_label[4]):
            class_label = 1
        elif int(class_label[5]):
            class_label = 2
        elif int(class_label[5]):
            class_label = 3
        else:
            class_label = 4
        bbox = []
        bbox_data = bbox_data.split(" ")
        label = []
        for i in range(0, len(bbox_data), 6):
            label.append(class_label)
            bbox.append([float(bbox_data[i + 2]), float(bbox_data[i + 3]), float(bbox_data[i + 4]), float(bbox_data[i + 5])])
        bbox = torch.as_tensor(bbox, dtype=torch.float32)
        labels = torch.as_tensor(label, dtype=torch.int64)
        area = (bbox[:, 3] - bbox[:, 1]) * (bbox[:, 2] - bbox[:, 0])
        image_id = torch.tensor([idx])
        target = {}
        target["boxes"] = bbox
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        if self.trans is not None:
            image, target = self.trans(image, target)
        return image, target
