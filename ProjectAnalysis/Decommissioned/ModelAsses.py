import argparse
import numpy as np
import pandas as pd
import os
import time

# def eval_randomsearch(clf, pgrid, xTrain, yTrain, xTest, yTest):
#     """
#     Given a sklearn classifier and a parameter grid to search,
#     choose the optimal parameters from pgrid using Random Search CV
#     and train the model using the training dataset and evaluate the
#     performance on the test dataset. The random search cv should try
#     at most 33% of the possible combinations.
#
#     Parameters
#     ----------
#     clf : sklearn.ClassifierMixin
#         The sklearn classifier model
#     pgrid : dict
#         The dictionary of parameters to tune for in the model
#     xTrain : nd-array with shape (n, d)
#         Training data
#     yTrain : 1d array with shape (n, )
#         Array of labels associated with training data
#     xTest : nd-array with shape (m, d)
#         Test data
#     yTest : 1d array with shape m
#         Array of labels associated with test data.
#
#     Returns
#     -------
#     resultDict: dict
#         A Python dictionary with the following 4 keys,
#         "AUC", "AUPRC", "F1", "Time" and the values are the floats
#         associated with them for the test set.
#     roc : dict
#         A Python dictionary with 2 keys, fpr, and tpr, where
#         each of the values are lists of the fpr and tpr associated
#         with different thresholds. You should be able to use this
#         to plot the ROC for the model performance on the test curve.
#     bestParams: dict
#         A Python dictionary with the best parameters chosen by your
#         GridSearch. The values in the parameters should be something
#         that was in the original pgrid.
#     """
#     start = time.time()
#
#     resultsDict = {"AUC": 0, "AUPRC": 0, "F1": 0, "Time": 0}
#     roc = {"fpr": [], "tpr": []}
#
#     totcombo = len(list(ParameterGrid(pgrid)))
#     n_iter = max(1, int(totcombo * 0.33))
#
#     grid = RandomizedSearchCV(clf, pgrid, n_iter=n_iter, scoring='f1', cv=5)
#     grid.fit(xTrain, yTrain)
#
#     best_model = grid.best_estimator_
#     best_params = grid.best_params_
#
#     timeElapsed = time.time() - start
#
#     yHat = best_model.predict(xTest)
#     yScore = best_model.predict_proba(xTest)[:, 1]
#
#     roc['fpr'], roc['tpr'], _ = roc_curve(yTest, yScore)
#
#     resultsDict["AUC"] = roc_auc_score(yTest, yHat)
#     resultsDict["AUPRC"] = average_precision_score(yTest, yHat)
#     resultsDict["F1"] = f1_score(yTest, yHat)
#     resultsDict["Time"] = timeElapsed
#
#     return resultsDict, roc, best_params


def _accuracy(yTrue, yHat):

    return np.sum(yHat == yTrue) / len(yTrue)