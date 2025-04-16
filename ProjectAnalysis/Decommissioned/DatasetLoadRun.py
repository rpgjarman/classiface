import pandas as pd
import pickle

from DatasetLoadMethods import *
from PreprocessingM import *

# Wiki Set:
if True:
    print("Loading all Wiki Data")
    ## Wiki Paths:
    set_path = '/Users/damienlo/Desktop/University/CS 334/Project/Datasets/Wiki'
    data_path = '/Users/damienlo/Desktop/University/CS 334/Project/Datasets/Wiki/WikiData'
    meta_path = '/Users/damienlo/Desktop/University/CS 334/Project/Datasets/Wiki/wiki.mat'
    struct_key = 'wiki'
    temp = '/Users/damienlo/Desktop/'


    # Handling Meta Data
    ## Creating MetaData CSV
    print("Creating Meta CSV")
    meta_csv_path = runAllMetaImports(set_path, meta_path, struct_key)
    print("Meta CSV creation completed")

    ## Loading metaData df
    print("Loading Meta Data as DF")
    meta_csv_df = getMetaDF(meta_csv_path)
    print("Loading meta_csv_df complete")

    ## Parsing Coords
    parse_coordinates(meta_csv_df)


    # Handling Image Data
    print("Loading Image List")
    meta_csv_df, img_list = importImageData(data_path, meta_csv_df)
    print("Image List Loaded, stacking to tensor")
    stacked_img_tensor = torch.stack(img_list)
    print("Image Tensor Stacked with dimentions:")
    print(stacked_img_tensor.size())

    print("Saving tensor to file and rewriting metacsv")
    saveTensor(stacked_img_tensor, temp)
    meta_csv_df.to_csv(meta_csv_path, index=False)
    print("Tensor Saved")


print("Program Complete")
