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
#=============================
#           META DATA
#=============================

'''
    convertDOB
    Adds column meta_data[data_year] 
'''
def convertDOB(meta_data):
    meta_data['birth_year'] = meta_data['dob'].apply(
        lambda x: (datetime.fromordinal(int(x)) + timedelta(days=x % 1) - timedelta(days=366)).year if x > 0 else None
    )

'''
    parse_coordinates
    Converts string od face_location to integer array
'''
def parse_coordinates(meta_data_df):
    meta_data_df['face_location'] = meta_data_df['face_location'].apply(
        lambda s: [float(num) for num in s.strip('[]').split()]
    )

'''
    calculateAge
    Calculate age from birth year
'''
def calculateAge(meta_data):
    convertDOB(meta_data)
    meta_data['age'] = meta_data['photo_taken'] - meta_data['birth_year']

'''
    handleMissingData
    Drops rows with no gender and fills average of age for rows with no age
'''
def handleMissingData(meta_data):
    meta_data = meta_data.dropna(subset=["gender"])

    mean_age = meta_data['age'].mean()
    meta_data['age'] = meta_data['age'].fillna(mean_age)

'''
    oneHotEncodeAgeBins
    One hot encodes age into 4 bins [-np.inf, mean - std, mean, mean + std, np.inf] into new meta_data['age_bin'] column
'''
def oneHotEncodeAgeBins(meta_data):
    mean = meta_data['age'].mean()
    std = meta_data['age'].std()
    bin_edges = [-np.inf, mean - std, mean, mean + std, np.inf]

    meta_data['age_bin'] = np.digitize(meta_data['age'], bins=bin_edges, right=False) - 1

    one_hot = pd.get_dummies(meta_data['age_bin'], prefix='age_bin')

    meta_data = pd.concat([meta_data, one_hot], axis=1)

'''
    dropIrrelevantColumns
    Return df with only meta_data['full_path','gender', 'age_bin']
'''
def dropIrrelevantColumns(meta_data):
    return meta_data[['full_path','gender', 'age_bin']]


#=============================
#          IMAGE DATA
#=============================
'''
importImageData
Given a meta_data_df, turns all images into tensors and returns list of cropped and resized tensors following df row order
Either a [[C,dimen,dimen]] if flatten = false or [C*dimen*dimen] if flatten = true 

Return: List of image tensors following row order of meta_data_df to be stacked

'''
def importImageData(data_path, meta_data_df, dimen, flatten):
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

        # Resize Image to dimen by dimen
        resize_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((dimen, dimen)),
            transforms.ToTensor()
        ])
        img_tensor = resize_transform(img_tensor)

        # Flatten Image to [C,H,W] -> C * H * W
        if flatten:
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


'''
    minMaxSaclerPixelValues
    Perform pixel by pixel min-max scaling for image data
'''
def minMaxSaclerPixelValues(trainX, testX):
    min_val = trainX.min(dim=0).values
    max_val = trainX.max(dim=0).values

    range_val = max_val - min_val
    range_val[range_val == 0] = 1e-6

    trainScaled =(trainX - min_val) / range_val
    testScaled = (testX - min_val) / range_val

    return trainScaled, testScaled

'''
    standardScalerPixelValues
    Perform pixel by pixel standardiation scaling for image data
'''
def standardScalerPixelValues(trainX, testX):
    mean = trainX.mean(dim=0)
    std = trainX.std(dim=0)
    std[std == 0] = 1e-6

    trainScaled = (trainX - mean) / std
    testScaled = (testX - mean) / std
    return trainScaled, testScaled


'''
    scaleImageFeatureValues
    Perform pixel by pixel scaling for image data specified type
    
    @Returns scaled tensors
'''
def scaleImageFeatureValues(trainX, testX, type):
    if type == 'standard':
        return standardScalerPixelValues(trainX, testX)

    if type == 'minmax':
        return minMaxSaclerPixelValues(trainX, testX)

    if type == 'ResNet':
        trainX = resNetScale(trainX)
        testX = resNetScale(testX)

        return trainX, testX

    else:
        print("Scaling Paramter does not exist")

'''
    resNetScale
    Apply image wise scaling by resnet mean and standard deviation
'''
def resNetScale(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
    return (tensor - mean) / std

