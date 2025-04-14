import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def standard_scale(xTrain, xTest):
    """
    Preprocess the training data to have zero mean and unit variance.
    The same transformation should be used on the test data. For example,
    if the mean and std deviation of feature 1 is 2 and 1.5, then each
    value of feature 1 in the test set is standardized using (x-2)/1.5.

    Parameters
    ----------
    xTrain : numpy.nd-array with shape (n, d)
        Training data 
    xTest : nd-array with shape (m, d)
        Test data 

    Returns
    -------
    xTrain : nd-array with shape (n, d)
        Transformed training data with mean 0 and unit variance 
    xTest : nd-array with  shape (m, d)
        Transformed test data using same process as training.
    """
    scaler = StandardScaler()
    xTrain = scaler.fit_transform(xTrain)
    xTest = scaler.transform(xTest)
    return xTrain, xTest


def minmax_range(xTrain, xTest):
    """
    Preprocess the data to have minimum value of 0 and maximum
    value of 1.T he same transformation should be used on the test data.
    For example, if the minimum and maximum of feature 1 is 0.5 and 2, then
    then feature 1 of test data is calculated as:
    (1 / (2 - 0.5)) * x - 0.5 * (1 / (2 - 0.5))

    Parameters
    ----------
    xTrain : numpy.nd-array with shape (n, d)
        Training data 
    xTest : nd-array with shape (m, d)
        Test data 

    Returns
    -------
    xTrain : nd-array with shape (n, d)
        Transformed training data with min 0 and max 1.
    xTest : nd-array with  shape (m, d)
        Transformed test data using same process as training.
    """
    scaler = MinMaxScaler()
    xTrain = scaler.fit_transform(xTrain)
    xTest = scaler.transform(xTest)
    return xTrain, xTest


def add_irr_feature(xTrain, xTest):
    """
    Add 2 features using Gaussian distribution with 0 mean,
    standard deviation of 50.

    Parameters
    ----------
    xTrain : nd-array with shape n x d
        Training data 
    xTest : nd-array with shape m x d
        Test data 

    Returns
    -------
    xTrain : nd-array with shape n x (d+2)
        Training data with 2 new noisy Gaussian features
    xTest : nd-array with shape m x (d+2)
        Test data with 2 new noisy Gaussian features
    """
    np.random.seed(None) 
    
    nTrain, _ = xTrain.shape
    nTest, _ = xTest.shape

    rand_train = np.random.normal(loc=0, scale=50, size=(nTrain, 2))
    rand_test = np.random.normal(loc=0, scale=50, size=(nTest, 2))
    
    xTrain = np.hstack((xTrain, rand_train))
    xTest = np.hstack((xTest, rand_test))
    
    return xTrain, xTest
