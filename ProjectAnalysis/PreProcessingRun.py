import pandas as pd
import pickle

from DatasetLoadMethods import *
from PreprocessingMethods import *

set_path = '/Users/damienlo/Desktop/University/CS 334/Project/Datasets/Wiki'
data_path = '/Users/damienlo/Desktop/University/CS 334/Project/Datasets/Wiki/WikiData'
meta_path = '/Users/damienlo/Desktop/University/CS 334/Project/Datasets/Wiki/wiki.mat'

testY = '/Users/damienlo/Desktop/University/CS 334/Project/Datasets/Wiki/WikiData/test_meta.csv'
testX = '/Users/damienlo/Desktop/University/CS 334/Project/Datasets/Wiki/WikiData/test_img_tensor.pt'

trainY = '/Users/damienlo/Desktop/University/CS 334/Project/Datasets/Wiki/WikiData/train_meta.csv'
trainX = '/Users/damienlo/Desktop/University/CS 334/Project/Datasets/Wiki/WikiData/train_img_tensor.pt'

# Loading In Data
print("Loading meta data and img tensors")
trainX = loadTensor(trainX)
testX = loadTensor(testX)

trainY = getMetaDF(trainY)
testY = getMetaDF(testY)
print("Meta Data and Tensors Loaded")

trainY = trainY[["gender", "age_bin"]]
testY = testY[["gender", "age_bin"]]

print("Begining Image PreProcessing")
# Image Feature Value Preprocessing
trainX, testX = scaleImageFeatureValues(trainX, testX, "standard")

print("Saving Preprocessed Data")
# Download Files
torch.save(trainX, '/Users/damienlo/Desktop/University/CS 334/Project/Datasets/ProcessedData/trainX.pt')
torch.save(testX, '/Users/damienlo/Desktop/University/CS 334/Project/Datasets/ProcessedData/testX.pt')

trainY.to_csv('/Users/damienlo/Desktop/University/CS 334/Project/Datasets/ProcessedData/trainY.csv')
testY.to_csv('/Users/damienlo/Desktop/University/CS 334/Project/Datasets/ProcessedData/testY.csv')




print("Program Complete")