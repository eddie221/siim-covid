#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 16:17:58 2021

@author: mmplab603
"""

import pandas as pd
import pydicom
import numpy as np
import os
import glob
import tqdm

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

bbox_file = pd.read_csv("../../datasets/siim-covid19-detection/train_image_level.csv")
study_file = pd.read_csv("../../datasets/siim-covid19-detection/train_study_level.csv")
study_file['StudyInstanceUID'] = study_file['id'].apply(lambda x : x[:-6])
merge_data = bbox_file.merge(study_file, on='StudyInstanceUID')
merge_data['ImageID'] = merge_data['id_x'].apply(lambda x : x[:-6])


bboxes = []
for i in tqdm.tqdm(range(len(merge_data))):
    study_id = merge_data.iloc[i]['StudyInstanceUID']
    image_id = merge_data.iloc[i]['ImageID']
    image_path = glob.glob(os.path.join("../../datasets/siim-covid19-detection/train/", study_id, "*", "{}.dcm".format(image_id)))[0]
    image = load_image(image_path)
    W, H = image.shape
    bbox = merge_data.iloc[i]["label"].split(" ")
    print(bbox)
    
    for i in range(0, len(bbox), 6):
        if bbox[i] == 'opacity':
            print((float(bbox[i + 2]) + float(bbox[i + 4]) / 2))
            print(W)
            bboxes.append([(float(bbox[i + 2]) + float(bbox[i + 4])) / 2 / W, 
                           (float(bbox[i + 3]) + float(bbox[i + 5])) / 2 / H, 
                           (float(bbox[i + 4]) - float(bbox[i + 2])) / W, 
                           (float(bbox[i + 5]) - float(bbox[i + 3])) / H])
    
            
            
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plt.imshow(image)
    for i in range(len(bboxes)):
        rect = patches.Rectangle(((bboxes[i][0] - bboxes[i][2] / 2) * W, (bboxes[i][1] - bboxes[i][3] / 2) * H),
                                 bboxes[i][2] * W,
                                 bboxes[i][3] * H,
                                 linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)
    break
        
# =============================================================================
#         print(bbox)
#         bboxes.append([bbox['x'] + bbox['width'] / 2,
#                        bbox['y'] + bbox['height'] / 2,
#                        bbox['width'],
#                        bbox['height']])
# =============================================================================
print(bboxes)