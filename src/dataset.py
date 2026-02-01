"""
Computer Vision Cats vs Dogs Project
dataset.py
Contains class ImageDataset
By: Kyle Webster

"""

from pathlib import Path # for file path
from PIL import Image # for images
import warnings

from torchvision.io import decode_image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, rootDir, transform = None): # initialze instance of class
        """
        Initialization method to create instance of ImageDataset
        
        Arugments:
            self: reference to self
            rootDir: directory path from root with dataset
            transform: function to transform image for retrieval

        Returns:
            None

        """
        self.rootDir = rootDir # directory path from root with image dataset
        self.transform = transform # transform function to transform image
        self.imgPathsAndLabels = [] # will have all image paths loaded numerically starting with all the cats and then the dogs.

        self.classAndLabel = { "Cat" : 0, "Dog" : 1 }

        for className, label in self.classAndLabel.items(): # loop through the directories storing data ("Cat" and "Dog" directories)
            classDir = Path(self.rootDir) / className
            
            for imgPath in classDir.iterdir(): # loop through each file in directory
                if imgPath.suffix in {".jpg", ".jpeg", ".png"}:
                    try: # to make sure image file is not corrupted
                        with Image.open(imgPath) as img:
                            img.verify()
                        self.imgPathsAndLabels.append((imgPath, label))
                    except Exception:
                        print(f"Removing corrupted image from dataset: {imgPath}")

    def __len__(self): # get length of dataset
        """
        Returns length of dataset in instance of ImageDataset
        
        Arguments:
            self: reference to self

        Returns:
            length of dataset in self

        """
        return len(self.imgPathsAndLabels)
        
    
    def __getitem__(self, idx): # get image and label in dataset of specified index
        """
        Gets item in dataset at specified index
        
        Arguments:
            self: reference to self
            idx: index that data is to be retrieved from

        Returns:
            PIL image of given index, classification label of image, path the image is in directory
            or None if image is corrupt

        """
        if idx >= self.__len__():
            raise ValueError("Index is out of range of dataset")
        
        imgPath, label = self.imgPathsAndLabels[idx]

        try: # to make sure image loads properly
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                image = Image.open(imgPath).convert("RGB") #decode_image(imgPath) - to tensor
        
            if self.transform:
                image = self.transform(image)

            return image, label, imgPath.name
        except Exception: # returns none if image is corrupt
            return None