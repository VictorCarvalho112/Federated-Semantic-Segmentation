from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path

import numpy as np
from torch import from_numpy

eval_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

map_classes = {
  7: "road",  # 0
  8: "sidewalk",  # 1
  11: "building",  # 2
  12: "wall",  # 3
  13: "fence",  # 4
  17: "pole",  # 5
  19: "light",  # 6
  20: "sign",  # 7
  21: "vegetation",  # 8
  22: "terrain",  # 9
  23: "sky",  # 10
  24: "person",  # 11
  25: "rider",  # 12
  26: "car",  # 13
  27: "truck",  # 14
  28: "bus",  # 15
  31: "train",  # 16
  32: "motorcycle",  # 17
  33: "bicycle"  # 18
} 



class GTA5(VisionDataset): # creating the dataset class (similarly to HM3)
    def __init__(self,
                img_dir,
                labels_dir,
                annotation_file, # annotation file directry that contains the files names to be used for the training
                target_transform = None,
                fda_transform = None,
                transform=None,
                cl19:bool = False):  # the train input defines if
        # the dataset will be partitioned for train or for test
        
        self.img_dir = img_dir
        self.labels_dir = labels_dir
        self.annotation_file = annotation_file
        self.transform = transform
        self.target_transform = target_transform
        self.fda_transform = fda_transform
        # image file name: 08688.png
        # the label file has the same name but it is stored in a different folder - 'labels' folder
        with open(annotation_file,'r') as f:
            image_files = f.readlines()

        for file in image_files:
            name = image_files[image_files.index(file)].replace('\n','')
            image_files[image_files.index(file)] = name
        
        self.images = image_files
        self.labels = image_files # same files but different directory 

        ### CHECK THE MAPPING AFTERWODS
        if cl19 and target_transform is None:
            classes = eval_classes
            mapping = np.zeros((256,), dtype=np.int64) + 255
            for i, cl in enumerate(classes):
                mapping[cl] = i
            self.target_transform = lambda x: from_numpy(mapping[x])
        
            
        
        
    def __getitem__(self, index):

        
        
        # images and labels from the proper directory
        img_path = os.path.join(self.img_dir,self.images[index]) # getting the image path from the given index (it will search from the list outputed at
        # the 'self.images' variable)
        label_path = os.path.join(self.labels_dir,self.labels[index]) # same as before but with the label

        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)

        image = image.resize( (1914,1052 ), Image.BICUBIC)
        label = label.resize( (1914,1052 ), Image.BICUBIC)

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image, label = self.transform(image,label)
            # label = self.transform(label)

        if self.target_transform is not None:
            label = self.target_transform(label)
    
        if self.fda_transform is not None:
            image = self.fda_transform(image)
        
        return image, label

    def __len__(self): # len definition is mandatory. Retunrs the length of the dataset
        length = len(self.images)
        return length