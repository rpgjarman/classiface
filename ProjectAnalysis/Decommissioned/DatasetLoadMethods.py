import os
import numpy as np
import pandas as pd
import scipy.io as sio
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from torchvision import transforms
import torch


# META DATA
'''
loadMetaDF
Loads the Metadata from .mat form into pandas dataframe sorted by path name and sort by full_path

Returns: pd.DataFrame
'''
def loadMetaDF(meta_path, struct_key):
    meta_data = sio.loadmat(meta_path)[struct_key][0][0]
    fields = [
        'full_path',
        'dob',
        'photo_taken',
        'gender',
        'face_score',
        'second_face_score',
        'face_location'
    ]

    data = {}
    for field in fields:
        value = meta_data[field][0]

        # Handle nested strings (paths)
        if field == 'full_path':
            data[field] = [v[0] for v in value]
        # Gender/score/dob/years are numeric arrays
        else:
            data[field] = value

    return pd.DataFrame(data).sort_values(by=['full_path'], ascending=True)

'''
downloadMetaCSV
Takes metdadata as dataframe and download it as csv to set_path
'''
def downloadMetaCSV(meta_df, set_path, struct_key):
    meta_df.to_csv(set_path + f'/{struct_key}_metadata.csv', index=False)

'''
runAllMetaImports
Given metadata .mat file path, creates a csv if not created already and returns csv path

Returns: path of CSV file
'''
def runAllMetaImports(set_path, meta_path, struct_key):
    if os.path.exists(set_path + f'/{struct_key}_metadata.csv'):
        print('Metadata already downloaded.')
    else:
        df = loadMetaDF(meta_path, struct_key)
        downloadMetaCSV(df, set_path, struct_key)
        print('Metadata downloaded.')
    return set_path + f'/{struct_key}_metadata.csv'

'''
getMetaDF
Get df of metaData from exisitng csv
'''
def getMetaDF(meta_csv_path):
    return pd.read_csv(meta_csv_path)


# IMAGE DATA
'''
createProcessDataDirectory
Create directory if not already exists for process data copying all subdirecftories [00:99] to math metadata path

Return: Path of process image main directory
'''
def createProcessDataDirectory(set_path, struct_key):
    path = set_path + f'/Processed{struct_key.capitalize()}Data'
    if not os.path.exists(path):
        os.makedirs(path)

        for i in range(0,10):
            os.makedirs(path+f'/0{i}')
        for i in range(10,100):
            os.makedirs(path+f'/{i}')

        print("Directory Created")

        return path

    print("Directory Already Exists")

    return path

'''
saveCroppedImages
Given 4D tensor, saves a jpg of the image in the appropaite folder
'''
def saveCroppedImages(stacked_img_tensor, meta_data_df, processed_data_path):
    for i, img_tensor in enumerate(stacked_img_tensor):
        # Define Specific Image Path
        path = os.path.join(processed_data_path, meta_data_df['full_path'].iloc[i])
        # Save Image
        to_pil = transforms.ToPILImage()
        img_pil = to_pil(img_tensor)
        img_pil.save(path)

'''
saveTensor
Saves the full 4D tenosr of [N,C,H,W] as a pt file to be loaded

Return: tensor Path
'''
def saveTensor(tensor, data_path,):
    path = data_path + "/full_tensor_data.pt"
    torch.save(tensor, path)
    return path

'''
loadTensor
Loads tensor given path

Return: tensor 
'''
def loadTensor(tensor_path):
    return torch.load(tensor_path)
