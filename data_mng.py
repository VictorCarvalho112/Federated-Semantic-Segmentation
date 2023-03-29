from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import math

import numpy as np
import pandas as pd
from torch import from_numpy

max_img_per_client = 20

### this file has functions and classes related to data management, so
# there are functions to open images, split the image list.


### function to create list with file names according to train/test partition A

def _dataset_info_A(img_folder_path, train=True): # this function outputs a list with the image files 
    # names, and another list with the label files names. The inputs are the folder paths of each and a boolean to retrieve
    # a list of train images/labels or test images/labels.

    img_files = os.listdir(img_folder_path) # getting a list of all the files in the specified directory

    # img file name 'aachen_000001_000019_leftImg8bit.png'
    # label file name 'aachen_000001_000019_gtFine_labelIds.png'

    img_dict_test = {} # creating a empty dictionary for retrieving the test image files names 

    for name in img_files:
        city_name = name.split('_')[0] # retrieving only the name of the city 

        if city_name not in img_dict_test.keys():
            img_dict_test[city_name] = [name] # adding the city name to the dictionary if it doesnt already existis
        elif city_name in img_dict_test.keys() and len(img_dict_test[city_name])<2:
            img_dict_test[city_name].append(name) # adding file with the city name until it is equal to 2 the number
            # of files with the same city name

    # creating the test lists
    img_test_files = np.array(list(img_dict_test.values())).reshape((1,-1)).tolist()[0]
    
    # creating the train lists by removing the test files of the original files lists
    img_train_files = img_files

    for file in img_test_files:
        img_train_files.remove(file)

    if train:
        return img_train_files
    else:
        return img_test_files


### function to create list with file names according to train/test partition B

def _dataset_info_B(annotation_file): # function to create a list with the file names. The input
    # is the txt file with the file names
    with open(annotation_file, 'r') as t: # reading the files and storing it in a list
        image_files = t.readlines()
    

    # img file name in txt file: strasbourg/strasbourg_000000_017593_leftImg8bit.png
    # label file name desired as output: strasbourg_000000_017593_gtFine_labelIds.png

    for name in image_files:
        name1 = image_files[image_files.index(name)].replace('\n','') # removing the '\n' string present in every line from the function 'readlines'
        name2 = name1.split('/')[1] # removing the first city name from the file 
        image_files[image_files.index(name)] = name2
    
    return image_files

def _dataset_info_GTA(annotation_file): # function to create a list with the file names. The input
    # is the txt file with the file names
    with open(annotation_file, 'r') as t: # reading the files and storing it in a list
        image_files = t.readlines()
    

    for name in image_files:
    
        image_files[image_files.index(name)] = name
    
    return image_files
### function to create uniform split as described in step 3

def uniform_split(dataset,num_clients):
    '''function to create the homogeneous split of a dataset, according to the number of clients (as described in step 3)
    Inputs: 
        dataset'''
    img_list = dataset.images
    img_dict = {}
    client_list = []

    for file in img_list:
        city = file.split('_')[0]
        if city not in img_dict.keys():
            img_dict[city] = []
        img_dict[city].append(file)
    
    clients_per_city = math.ceil(num_clients/len(img_dict.keys()))

    for i,city in enumerate(img_dict.keys()):
        x = np.array_split(img_dict[city],clients_per_city)
        y = [list(array) for array in x]
        for j in y:
            client_list.append(j)
        
    for i in range(len(client_list)):
        if len(client_list[i])>max_img_per_client:
            del client_list[i][20:]
        
    if len(client_list)>num_clients:
        client_list = client_list[:num_clients]

    return client_list

    

def heterogeneous_split(dataset,num_clients):
    '''function to create heterogeneous split as described in step 3
        Inputs: 
            dataset: dataset partition from step 1, (cityscapes dataset)
            num_clients: int, numeber of desired clients
        Output: 
            client_list: 2D list, where 1st index is each client and 2nd index is image file'''

    img_list = dataset.images

    img_list = np.array(img_list)
    np.random.shuffle(img_list)
    client_array = np.array_split(img_list,num_clients)
    client_list = [list(array) for array in client_array]

    for i in range(len(client_list)):
        if len(client_list[i])>max_img_per_client:
            del client_list[i][20:]
    
    return client_list
    


