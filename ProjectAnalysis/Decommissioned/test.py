from DatasetLoadMethods import *
from PreprocessingM import *
import torch


RES_trainX1 = loadTensor('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/stacked_tensor_data1.pt')
RES_trainX2 = loadTensor('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/stacked_tensor_data2.pt')
RES_trainX3 = loadTensor('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/stacked_tensor_data3.pt')
RES_trainX4 = loadTensor('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/stacked_tensor_data4.pt')

RES_testX = loadTensor('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/stacked_tensor_data5.pt')


print(f"Train Package 1 Size: {RES_trainX1.shape}")
print(f"Train Package 2 Size: {RES_trainX2.shape}")
print(f"Train Package 3 Size: {RES_trainX3.shape}")
print(f"Train Package 4 Size: {RES_trainX4.shape}")

print(f"Test Package Size: {RES_testX.shape}")
