import pandas as pd
import numpy as np
import argparse
from src.knn import Knn, mean_absolute_error, r_squared
from src.DataImportMethods import runAllMetaImports, getMetaDF, load_image_data
from sklearn.model_selection import train_test_split

# Paths
meta_path = "./wiki/wiki.mat"
image_dir = "./wiki/wiki_crop"
meta_csv_path = "./wiki/meta.csv"
struct_key = "wiki"

# Output CSVs
xTrain_path = "./wiki/xTrain.csv"
yTrain_path = "./wiki/yTrain.csv"
xTest_path = "./wiki/xTest.csv"
yTest_path = "./wiki/yTest.csv"

# 1. Import metadata and convert .mat to .csv if needed
runAllMetaImports(image_dir, meta_path, struct_key, meta_csv_path)
df = getMetaDF(meta_csv_path)

# 2. Convert image data to NumPy arrays
x_data, y_data = load_image_data(df, image_dir)
df = df.iloc[:len(x_data)].copy()  # Ensure metadata matches image count

# 3. Split and save data
def splitData(meta_data, image_array, train_split, xTrain_path, yTrain_path, xTest_path, yTest_path):
    if 'age' not in meta_data.columns:
        meta_data['age'] = meta_data['photo_taken'] - (meta_data['dob'] / 365.25 + 1969)

    x_train, x_test, y_train, y_test = train_test_split(
        image_array, meta_data['age'].values, test_size=(1 - train_split), random_state=42)

    pd.DataFrame(x_train).to_csv(xTrain_path, index=False)
    pd.DataFrame(x_test).to_csv(xTest_path, index=False)
    pd.DataFrame(y_train, columns=['age']).to_csv(yTrain_path, index=False)
    pd.DataFrame(y_test, columns=['age']).to_csv(yTest_path, index=False)

splitData(df, x_data, 0.8, xTrain_path, yTrain_path, xTest_path, yTest_path)

# 4. Load the split data
xTrain = pd.read_csv(xTrain_path).to_numpy()
yTrain = pd.read_csv(yTrain_path).to_numpy().flatten()
xTest = pd.read_csv(xTest_path).to_numpy()
yTest = pd.read_csv(yTest_path).to_numpy().flatten()

# 5. Train kNN and evaluate
knn = Knn(k=5)
knn.train(xTrain, yTrain)
yHat = knn.predict(xTest)

print("Evaluation Metrics:")
print("-------------------")
print(f"MAE: {mean_absolute_error(yHat, yTest):.2f} years")
print(f"R^2: {r_squared(yHat, yTest):.3f}")
