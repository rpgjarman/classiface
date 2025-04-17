from src.load_data import load_csv_data
from src.split_and_scale import split_data
from src.knn import Knn, mean_absolute_error, r_squared
from joblib import dump

# 1. Load CSV
csv_path = "data/wiki_faces.csv"
x, y = load_csv_data(csv_path, target="age")

# 2. Train/Test Split + Standard Scaling
xTrain, xTest, yTrain, yTest = split_data(x, y)

# 3. Train kNN
# Main hyperparameter
knn = Knn(k=5)
knn.train(xTrain, yTrain)

# 4. Predict
yHat = knn.predict(xTest)

# 5. Evaluation
mae = mean_absolute_error(yHat, yTest)
r2 = r_squared(yHat, yTest)

print("Evaluation Metrics:")
print("-------------------")
print(f"MAE:  {mae:.2f} years")
print(f"R^2:  {r2:.3f}")

# 6. Save Model
dump(knn, "models/knn_model.pkl")
