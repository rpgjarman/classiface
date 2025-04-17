import torch

'''
    splitData
    Splits data into rounded train and test sets saves locally to given paths
'''
def splitData(meta_data, stacked_img_tensor, train_split, xTrain_path, yTrain_path, xTest_path, yTest_path):
    dataset_size = stacked_img_tensor.size()[0]
    train_size = round(dataset_size * train_split)
    test_size = dataset_size - train_size

    # Tensor Data
    train_img_tensor, test_img_tensor = torch.split(stacked_img_tensor, [train_size, test_size])
    torch.save(train_img_tensor,xTrain_path)
    torch.save(test_img_tensor,xTest_path)

    # Meta Data
    train_meta = meta_data.iloc[0:train_size]
    test_meta = meta_data.iloc[train_size:]
    train_meta.to_csv(yTrain_path,index=False)
    test_meta.to_csv(yTest_path,index=False)