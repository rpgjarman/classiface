import numpy as np
from sklearn.model_selection import KFold
from src.knn import Knn, mean_absolute_error

def find_best_k(x, y, k_values=[1, 3, 5, 7, 9, 11, 15], folds=5):
    best_k = None
    lowest_mae = float('inf')

    for k in k_values:
        print(f"\nEvaluating k = {k}")
        fold_maes = []

        kf = KFold(n_splits=folds, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(x):
            xTrain, xVal = x[train_idx], x[val_idx]
            yTrain, yVal = y[train_idx], y[val_idx]

            knn = Knn(k)
            knn.train(xTrain, yTrain)
            yPred = knn.predict(xVal)
            mae = mean_absolute_error(yPred, yVal)
            fold_maes.append(mae)

        avg_mae = np.mean(fold_maes)
        print(f"Average MAE for k = {k}: {avg_mae:.2f}")

        if avg_mae < lowest_mae:
            lowest_mae = avg_mae
            best_k = k

    print(f"\nBest k: {best_k} with MAE: {lowest_mae:.2f}")
    return best_k
