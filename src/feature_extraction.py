import numpy as np
from skimage.feature import hog
from PIL import Image
import os

def extract_hog_features(images, image_size=(64, 64)):
    """
    Extract HOG features from a set of images
    
    Parameters
    ----------
    images : numpy array or list of images
        The images to extract features from
    image_size : tuple (height, width)
        The size to resize images to before feature extraction
        
    Returns
    -------
    hog_features : numpy array
        The HOG features for each image
    """
    hog_features = []
    for image in images:
        img = Image.fromarray(image.reshape(image_size))
        features = hog(np.array(img), pixels_per_cell=(8, 8), 
                     cells_per_block=(2, 2), feature_vector=True)
        hog_features.append(features)
    return np.array(hog_features)

def extract_hog(image_path, image_size=(64, 64)):
    """
    Extract HOG features from a single image file
    
    Parameters
    ----------
    image_path : str
        Path to the image file
    image_size : tuple (height, width)
        The size to resize images to before feature extraction
        
    Returns
    -------
    features : numpy array or None
        The HOG features for the image, or None if the image couldn't be loaded
    """
    try:
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            return None
            
        img = Image.open(image_path).convert('L').resize(image_size)
        features = hog(np.array(img), pixels_per_cell=(8, 8), 
                     cells_per_block=(2, 2), feature_vector=True)
        return features
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None