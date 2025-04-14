import argparse
import numpy as np
import pandas as pd

# specific to solutions
from scipy import stats as st
import sklearn.metrics as skm
import sklearn.neighbors as skn
from scipy import stats

class Knn(object):
    k = 0              # number of neighbors to use
    nFeatures = 0      # number of features seen in training
    nSamples = 0       # number of samples seen in training
    isFitted = False  # has train been called on a dataset?
    ## below here is specific to solution 
    xFeat = None  # add storage of the features
    y = None      # add storage of the labels

    def __init__(self, k):
        """
        Knn constructor

        Parameters
        ----------
        k : int 
            Number of neighbors to use.
        """
        self.k = k

    def train(self, xFeat, y):
        """
        Train the k-nn model.

        Parameters
        ----------
        xFeat : numpy nd-array with shape (n, d)
            Training data 
        y : 1d array with shape (n, )
            Array of labels associated with training data.

        Returns
        -------
        self : object
        """
        # store the two objects
        self.xFeat = xFeat
        self.nFeatures = xFeat.shape[1]
        self.nSamples = xFeat.shape[0]
        self.y = y
        self.isFitted = True
        return self


    def predict(self, xFeat):
        yHat = []
        for i in range(xFeat.shape[0]):
            dist = np.linalg.norm(self.xFeat - xFeat[i, :], axis=1)
            idx = np.argsort(dist)
            yNeighbors = self.y[idx[:self.k]]
            yHat.append(np.mean(yNeighbors))  # regression = average of neighbors
            # yHat.append(stats.mode(yNeighbors)[0][0])
        return np.array(yHat)


def mean_absolute_error(yHat, yTrue):
    return np.mean(np.abs(yHat - yTrue))
    
def r_squared(yHat, yTrue):
    ss_res = np.sum((yTrue - yHat)**2)
    ss_tot = np.sum((yTrue - np.mean(yTrue))**2)
    return 1 - ss_res / ss_tot


def accuracy(yHat, yTrue):
    """
    Calculate the accuracy of the prediction

    Parameters
    ----------
    yHat : 1d-array with shape (n,)
        Predicted class label for n samples
    yTrue : 1d-array with shape (n, )
        True labels associated with the n samples

    Returns
    -------
    acc : float between [0,1]
        The accuracy of the model
    """
    # TODO calculate the accuracy
    acc = np.sum(yHat == yTrue) / len(yTrue)
    return acc
    # return np.mean(yHat == yTrue)


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("k",
                        type=int,
                        help="the number of neighbors")
    parser.add_argument("--xTrain",
                        default="simxTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="simyTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="simxTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="simyTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    # assume the data is all numerical and 
    # no additional pre-processing is necessary
    xTrain = pd.read_csv(args.xTrain).to_numpy()
    yTrain = pd.read_csv(args.yTrain).to_numpy().flatten()
    xTest = pd.read_csv(args.xTest).to_numpy()
    yTest = pd.read_csv(args.yTest).to_numpy().flatten()
    # create an instance of the model
    knn = Knn(args.k)
    knn.train(xTrain, yTrain)
    # predict the training dataset
    yHatTrain = knn.predict(xTrain)
    trainAcc = accuracy(yHatTrain, yTrain)
    # predict the test dataset
    yHatTest = knn.predict(xTest)
    testAcc = accuracy(yHatTest, yTest)
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)

    # This is a sanity check and not in the original file
    knn = skn.KNeighborsClassifier(args.k)
    knn.fit(xTrain, yTrain)
    yHatTrain = knn.predict(xTrain)
    yHatTest = knn.predict(xTest)
    trainAcc = skm.accuracy_score(yTrain, yHatTrain)
    testAcc = skm.accuracy_score(yTest, yHatTest)
    print("sklearn - Training Acc:", trainAcc)
    print("sklearn - Test Acc:", testAcc)

if __name__ == "__main__":
    main()
