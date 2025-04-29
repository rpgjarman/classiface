import os
import numpy as np
import pandas as pd
import scipy.io as sio
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision import transforms
import torch
from PIL import Image

# META DATA PREPROCESING
def convertDOB(meta_data):
    meta_data['birth_year'] = meta_data['dob'].apply(
        lambda x: (datetime.fromordinal(int(x)) + timedelta(days=x % 1) - timedelta(days=366)).year if x > 0 else None
    )

def parse_coordinates(meta_data_df):
    meta_data_df['face_location'] = meta_data_df['face_location'].apply(
        lambda s: [float(num) for num in s.strip('[]').split()]
    )

def calculateAge(data):
    data['age'] = data['photo_taken'] - data['birth_year']


# IMAGE PREPROCESSING
'''
importImageData
Given a meta_data_df, turns all images into tensors and returns list of cropped and resized tensors following df row order

Return: List of image tensors following row order of meta_data_df

'''
def importImageData(data_path, meta_data_df):
    data_list = []
    valid_indices = []

    for index, row in meta_data_df.iterrows():
        path = os.path.join(data_path, row['full_path'])

        # If the image does not exist (imdb set will only have some images in the meta data)
        if not os.path.exists(path):
            print(f'{row['full_path']} not found.')
            continue


        valid_indices.append(index)
        # Open image as tensor
        img = Image.open(path).convert('RGB')
        # Tensor colour scaled to [0:1]
        img_tensor = transforms.ToTensor()(img)

        # Crop Image
        img_tensor = cropFace(img_tensor, row['face_location'])
        # Resize Image to 128 by 128
        resize_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        img_tensor = resize_transform(img_tensor)

        # Flatten Image to [C,H,W] -> C * H * W
        img_tensor = img_tensor.view(-1)

        data_list.append(img_tensor)

    meta_data_df = meta_data_df.loc[valid_indices].reset_index(drop=True)
    return meta_data_df, data_list


'''
cropFace
Crops out face given croop coordinates

Returns: Cropped Face Image tensor
'''
def cropFace(img_tensor, crop_coords):
    left = round(crop_coords[0])
    top = round(crop_coords[1])
    right = round(crop_coords[2])
    bottom = round(crop_coords[3])

    # PyTorch image tensors: [C, H, W]
    return img_tensor[:, top:bottom, left:right]


def minMaxSaclerPixelValues(trainX, testX):
    min_val = trainX.min(dim=0).values
    max_val = trainX.max(dim=0).values

    range_val = max_val - min_val
    range_val[range_val == 0] = 1e-6

    trainScaled =(trainX - min_val) / range_val
    testScaled = (testX - min_val) / range_val

    return trainScaled, testScaled




def standardScalerPixelValues(trainX, testX):
    mean = trainX.mean(dim=0)
    std = trainX.std(dim=0)
    std[std == 0] = 1e-6

    trainScaled = (trainX - mean) / std
    testScaled = (testX - mean) / std
    return trainScaled, testScaled

def scaleImageFeatureValues(trainX, testX, type):
    if type == 'standard':
        return standardScalerPixelValues(trainX, testX)

    if type == 'minmax':
        return minMaxSaclerPixelValues(trainX, testX)

    else:
        print("Scaling Paramter does not exist")