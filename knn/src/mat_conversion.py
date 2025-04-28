import os
import scipy.io
import numpy as np
import pandas as pd
from PIL import Image

# Get repo root path (regardless of where script is run from)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths relative to the root
mat_path = os.path.join(ROOT_DIR, "data/wiki/wiki.mat")
img_root = os.path.join(ROOT_DIR, "data/wiki")
output_csv = os.path.join(ROOT_DIR, "data/wiki_faces.csv")

# Load .mat file
mat = scipy.io.loadmat(mat_path)
wiki = mat['wiki'][0, 0]

# Extract metadata
dob = wiki['dob'][0]
photo_taken = wiki['photo_taken'][0]
full_path = wiki['full_path'][0]
gender = wiki['gender'][0]
face_score = wiki['face_score'][0]
second_face_score = wiki['second_face_score'][0]
face_location = wiki['face_location'][0]  # (x1, y1, x2, y2)

# Estimate age
age = photo_taken - (dob / 365.25 + 1969)

# Filter valid images
valid = (face_score > 0) & np.isnan(second_face_score) & ~np.isnan(gender)

data = []

for i in np.where(valid)[0]:
    try:
        img_path = os.path.join(img_root, full_path[i][0])
        loc = face_location[i][0]

        # Crop to face
        img = Image.open(img_path).convert('L')
        cropped = img.crop((loc[0], loc[1], loc[2], loc[3]))  # (left, top, right, bottom)
        resized = cropped.resize((64, 64))
        pixels = np.asarray(resized).flatten()

        # Create one row: pixel_0, ..., pixel_4095, age, gender
        row = list(pixels) + [age[i], int(gender[i])]
        data.append(row)

    except Exception as e:
        continue

# Create DataFrame and column names
pixel_cols = [f"pixel_{i}" for i in range(64*64)]
df = pd.DataFrame(data, columns=pixel_cols + ["age", "gender"])

# Save as CSV
df.to_csv(output_csv, index=False)
print(f"Saved {len(df)} rows to {output_csv}")

# Output file has 4096 = grayscale pixel values, 1 = age, 1 = gender

