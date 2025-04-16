from DatasetLoadMethods import *
from PreprocessingM import *
import torch




stacked_img_tensor = loadTensor('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/stacked_tensor_data.pt')
meta_data = getMetaDF('/Users/damienlo/Desktop/clean_meta.csv')
train_split = 0.8

dataset_size = stacked_img_tensor.size()[0]
train_size = round(dataset_size * train_split)
test_size = dataset_size - train_size

# Splitting Tenor (Preserving Order)
train_img_tensor, test_img_tensor = torch.split(stacked_img_tensor, [train_size, test_size])
torch.save(train_img_tensor, '/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/train_img_tensor.pt')
torch.save(test_img_tensor, '/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/test_img_tensor.pt')

# Split MetaData Preserving Order only keep gender and age
train_meta = meta_data.iloc[0:train_size]
test_meta = meta_data.iloc[train_size:]
train_meta.to_csv('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/train_meta.csv', index=False)
test_meta.to_csv('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/test_meta.csv', index=False)

print(f'Train Tenosr size: {train_img_tensor.size()}')
print(f'Train meta size: {len(train_meta)}')
print(f'Test tenosr size: {test_img_tensor.size()}')
print(f'Test meta size: {len(test_meta)}')


print("Program Complete")