from pathlib import Path # for file path
from PIL import Image # for images

from torchvision.io import decode_image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, rootDir, transform = None): # initialze instance of class
        self.rootDir = rootDir # directory path from root with image dataset
        self.transform = transform # transform function to transform image
        self.imgPathsAndLabels = [] # will have all image paths loaded numerically starting with all the cats and then the dogs.

        self.classAndLabel = { "Cat" : 0, "Dog" : 1 }

        for className, label in self.classAndLabel.items(): # loop through the directories storing data ("Cat" and "Dog" directories)
            classDir = Path(self.rootDir) / className
            
            for imgPath in classDir.iterdir(): # loop through each file in directory
                if imgPath.suffix in {".jpg", ".jpeg", ".png"}:
                    self.imgPathsAndLabels.append((imgPath, label))

    def __len__(self): # get length of dataset
        return len(self.imgPathsAndLabels)
        
    
    def __getitem__(self, idx): # get image and label in dataset of specified index
        if idx >= self.__len__():
            raise ValueError("Index is out of range of dataset")
        
        imgPath, label = self.imgPathsAndLabels[idx]

        image = Image.open(imgPath) #decode_image(imgPath) - to tensor
        
        if self.transform:
            image = self.transform(image)

        return image, label