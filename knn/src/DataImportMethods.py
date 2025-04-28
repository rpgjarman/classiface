import os
import numpy as np
import pandas as pd
import scipy.io as sio
from PIL import Image

#=============================
#           META DATA
#=============================
def runAllMetaImports(set_path, meta_path, struct_key, meta_csv_path):
    if os.path.exists(meta_csv_path):
        print('Metadata already converted to CSV.')
    else:
        df = intialMetaDataLoad(meta_path, struct_key)
        downloadMetaCSV(df, meta_csv_path)
        print('Metadata downloaded.')
    return meta_csv_path

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
        if field == 'full_path':
            data[field] = [v[0] for v in value]
        else:
            data[field] = value
    return pd.DataFrame(data).sort_values(by=['full_path'], ascending=True)

def downloadMetaCSV(meta_df, meta_csv_path):
    meta_df.to_csv(meta_csv_path, index=False)

def getMetaDF(meta_csv_path):
    return pd.read_csv(meta_csv_path)

def load_image_data(df, image_dir, resize_shape=(64, 64)):
    x_data = []
    y_data = []
    valid_indices = []  # Keep track of which rows have valid images
    
    for i, row in df.iterrows():
        try:
            # Normalize relative path in case it includes 'wiki_crop/' already
            relative_path = row['full_path']
            if relative_path.startswith("wiki_crop/") or relative_path.startswith("wiki_crop\\"):
                relative_path = relative_path.split("/", 1)[-1]  # remove 'wiki_crop/' prefix
                
            # Ensure consistent path format with forward slashes
            clean_path = relative_path.replace("\\", "/")
            # Combine with base image_dir to get full path
            path = os.path.join(image_dir, clean_path)
            
            if not os.path.exists(path):
                print(f"File not found: {path}")
                continue
                
            age = row['photo_taken'] - (row['dob'] / 365.25 + 1969)
            img = Image.open(path).convert('L').resize(resize_shape)
            x_data.append(np.asarray(img).flatten())
            y_data.append(age)
            valid_indices.append(i)  # Store the index of this valid image
            
        except Exception as e:
            print(f"Error loading {row['full_path']}: {e}")
            continue
            
    print(f"Loaded {len(x_data)} valid images.")
    
    # Return the filtered dataframe containing only the rows with valid images
    filtered_df = df.iloc[valid_indices].copy() if valid_indices else df.iloc[0:0].copy()
    return np.array(x_data), np.array(y_data), filtered_df