import scipy.io
import numpy as np
import os
from PIL import Image
from sklearn.decomposition import PCA
import pandas as pd

def load_wiki_data(mat_path, image_dir, max_samples=None):
    mat = scipy.io.loadmat(mat_path)
    wiki = mat['wiki'][0, 0]

    dob = wiki['dob'][0]
    photo_taken = wiki['photo_taken'][0]
    full_path = wiki['full_path'][0]
    gender = wiki['gender'][0]
    face_score = wiki['face_score'][0]
    second_face_score = wiki['second_face_score'][0]

    age = photo_taken - (dob / 365.25)  # matlab serial to age estimate

    # Filter: good face score, no second face
    valid = (face_score > 0.0) & np.isnan(second_face_score) & ~np.isnan(gender)

    x_data = []
    y_data = []

    for i in np.where(valid)[0][:max_samples]:  # slice for dev/testing
        path = os.path.join(image_dir, full_path[i][0])  # Handles .jpg paths
        try:
            # Grayscale scaling, flattened pixel-by-pixel
            img = Image.open(path).convert('L').resize((64, 64))
            x_data.append(np.asarray(img).flatten())  # 64x64 = 4096 features
            y_data.append(age[i])
            # y_data.append(int(gender[i]))  # 0 or 1
        except:
            continue
    
    # Convert to numpy arrays
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    pca = PCA(n_components=100)
    x_data = pca.fit_transform(x_data)
    
    return x_data, y_data

def load_csv_data(csv_path, target="age"):
    """
    Load preprocessed image data from CSV and return features and labels.

    Parameters
    ----------
    csv_path : str
        Path to the wiki_faces.csv file
    target : str ("age" or "gender")
        Target column to use for labels

    Returns
    -------
    x : ndarray
        Feature matrix of shape (n_samples, 4096)
    y : ndarray
        Labels (age or gender) of shape (n_samples,)
    """
    df = pd.read_csv(csv_path)

    if target not in ["age", "gender"]:
        raise ValueError("Target must be either 'age' or 'gender'.")

    # Feature selection
    # Model input data is the pixel values from the images
    x = df.drop(columns=["age", "gender"]).values
    y = df[target].values

    return x, y