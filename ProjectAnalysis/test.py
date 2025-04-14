
import pandas as pd
from torchvision import transforms
import torch
from PIL import Image
import os

set_path = '/Users/damienlo/Desktop/University/CS 334/Project/Datasets/Wiki'
data_path = '/Users/damienlo/Desktop/University/CS 334/Project/Datasets/Wiki/WikiData'
meta_path = '/Users/damienlo/Desktop/University/CS 334/Project/Datasets/Wiki/wiki.mat'
struct_key = 'wiki'
meta_csv_path = ('/Users/damienlo/Desktop/University/CS 334/Project/Datasets/Wiki/wiki_metadata.csv')






def importImageData(data_path, meta_data_df):
    data_list = []

    for index, row in meta_data_df.iterrows():
        path = os.path.join(data_path, row['full_path'])

        # Open image as tensor
        img = Image.open(path).convert('RGB')
        # Tensor colour scaled to [0:1]
        img_tensor = transforms.ToTensor()(img)

        # Crop Image
        img_tensor = cropFace(img_tensor, row['face_location'])

        # Resize Image to 128 by 128
        resize_transform = transforms.Compose([
            transforms.ToPILImage(),  # tensor → PIL
            transforms.Resize((128, 128)),  # resize to 128x128
            transforms.ToTensor()  # PIL → tensor
        ])
        img_tensor = resize_transform(img_tensor)

        to_pil = transforms.ToPILImage()
        img_pil = to_pil(img_tensor)

        # Save to file
        img_pil.save('/Users/damienlo/Desktop/University/CS 334/Project/Datasets/Wiki/test.jpg')




def cropFace(img_tensor, crop_coords):
    left = round(crop_coords[0])
    top = round(crop_coords[1])
    right = round(crop_coords[2])
    bottom = round(crop_coords[3])

    # PyTorch image tensors: [C, H, W]
    return img_tensor[:, top:bottom, left:right]


def parse_coordinates(meta_data_df):
    meta_data_df['face_location'] = meta_data_df['face_location'].apply(
        lambda s: [float(num) for num in s.strip('[]').split()]
    )

meta_data_df = pd.read_csv(meta_csv_path).iloc[[0]]

parse_coordinates(meta_data_df)

importImageData(data_path, meta_data_df)
print("DONE")