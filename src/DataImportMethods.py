import os
import numpy as np
import pandas as pd
import scipy.io as sio
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from torchvision import transforms
import torch


#=============================
#           META DATA
#=============================
'''
runAllMetaImports
Given metadata .mat file path, creates a csv if not created already and returns csv path

Returns: path of CSV file
'''
def runAllMetaImports(set_path, meta_path, struct_key, meta_csv_path):
    if os.path.exists(meta_csv_path):
        print('Metadata already converted to CSV.')
    else:
        df = intialMetaDataLoad(meta_path, struct_key)
        downloadMetaCSV(df, meta_csv_path)
        print('Metadata downloaded.')
    return meta_csv_path

'''
intialMetaDataLoad
Loads the Metadata from .mat form into pandas dataframe sorted by path name and sort by full_path

Returns: pd.DataFrame
'''
def intialMetaDataLoad(meta_path, struct_key):
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
def downloadMetaCSV(meta_df, meta_csv_path):
    meta_df.to_csv(meta_csv_path, index=False)

'''
getMetaDF
Get df of metaData from exisitng csv
'''
def getMetaDF(meta_csv_path):
    return pd.read_csv(meta_csv_path)

#=============================
#          IMAGE DATA
#=============================

'''
saveTensor
Saves the full as a pt file to be loaded

Return: tensor Path
'''
def saveTensor(tensor, data_path):
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
