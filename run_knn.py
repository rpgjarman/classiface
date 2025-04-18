import pandas as pd
import numpy as np
from src.knn import Knn, mean_absolute_error, r_squared
from src.DataImportMethods import runAllMetaImports, getMetaDF, load_image_data
from src.DataSplitMethods import splitData

# === File Paths ===
meta_path = "data/wiki/wiki.mat"
image_dir = "data/wiki/wiki_crop"
meta_csv_path = "data/wiki/meta.csv"
struct_key = "wiki"

xTrain_path = "data/wiki/xTrain.csv"
yTrain_path = "data/wiki/yTrain.csv"
xTest_path = "data/wiki/xTest.csv"
yTest_path = "data/wiki/yTest.csv"

# === Step 1: Convert .mat to .csv ===
runAllMetaImports(image_dir, meta_path, struct_key, meta_csv_path)
df = getMetaDF(meta_csv_path)

# === Step 2: Load images as NumPy array and align metadata ===
x_data, y_data = load_image_data(df, image_dir)
df = df.iloc[:len(x_data)].copy()

# === Step 3: Split and save ===
splitData(df, x_data, 0.8, xTrain_path, yTrain_path, xTest_path, yTest_path)

# === Step 4: Load the split data ===
xTrain = pd.read_csv(xTrain_path).to_numpy()
yTrain = pd.read_csv(yTrain_path).to_numpy().flatten()
xTest = pd.read_csv(xTest_path).to_numpy()
yTest = pd.read_csv(yTest_path).to_numpy().flatten()

# === Step 5: Train and evaluate kNN ===
knn = Knn(k=5)
knn.train(xTrain, yTrain)
yHat = knn.predict(xTest)

print("Evaluation Metrics:")
print("-------------------")
print(f"MAE: {mean_absolute_error(yHat, yTest):.2f} years")
print(f"R^2: {r_squared(yHat, yTest):.3f}")
