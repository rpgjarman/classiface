from src.load_data import load_wiki_data
from src.split_and_scale import split_data
from src.knn import Knn, mean_absolute_error, r_squared
from joblib import dump

mat_path = "data/wiki/wiki.mat"
img_dir = "data/wiki"
x, y = load_wiki_data(mat_path, img_dir, max_samples=1000)  # start small

xTrain, xTest, yTrain, yTest = split_data(x, y)

knn = Knn(k=5)
knn.train(xTrain, yTrain)
yHat = knn.predict(xTest)

print("MAE:", mean_absolute_error(yHat, yTest))
print("R^2:", r_squared(yHat, yTest))

dump(knn, "models/knn_model.pkl")

# knn = joblib.load("models/best_knn_model.pkl")
