from dataset import ImageDataset
from pathlib import Path # for file paths 
import matplotlib.pyplot as plt # for graph plotting and displaying
import numpy as np # for array manipulation

from torch import Generator, device
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader # random_split - to split dataset into train and val | Dataloader - to create batches of dataset
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

#constants
ROOT = Path(__file__).parent / ".."

IMG_SIZE = 256 # height and width to change image to
CROP_SIZE = 224
TRAIN_SPLIT = 0.8 # percentage to split the data by
BATCH_SIZE = 64 # number of data samples being used in each batch
WORKERS = 0 # number of parallel processes for loading data in

SEED = 18
gen = Generator()
gen.manual_seed(SEED)

# obj of transform used to transform images when loaded in with given size, to a tensor and given normals
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transformOBJ = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), 
                                   transforms.CenterCrop(CROP_SIZE), 
                                   transforms.ToTensor(), 
                                   transforms.Normalize(mean, std)])

def createModel(): # creates ResNet18 model with 2 output classification layer
    model = resnet18(weights = ResNet18_Weights.DEFAULT) # create resnet18 model
    inFeatures = model.fc.in_features
    model.fc = nn.Linear(inFeatures, 2) # change the classifier layer to 2 outputs
    device = device("cuda" if torch.cuda.is_available() else "cpu") # will use gpu if available
    model.to(device)

    return model


def main():
    # --------------
    # Task 1
    # --------------
    datasetDir = ROOT / "data"
    dataset = ImageDataset(datasetDir, transformOBJ)
    print(f"Length of dataset: {len(dataset)}")
    image, label = dataset[3]
    print(type(image))
    if label == 0:
        print("Type: Cat")
    elif label == 1:
        print("Type: Dog")

    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.title("Image 4 from dataset")
    plt.show()

    _, height, width = image.shape
    print(f"Image has width: {width}px and height: {height}px")
    
    # --------------
    # Task 2
    # --------------
    trainSize = int(len(dataset) * TRAIN_SPLIT)
    valSize = len(dataset) - trainSize

    trainDataset, valDataset = random_split(dataset, [trainSize, valSize], gen)

    trainLoader = DataLoader(trainDataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = WORKERS)
    valLoader = DataLoader(valDataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = WORKERS)

    #checking to make sure loaders can be iterated
    batchOne = next(iter(trainLoader))
    batchTwo = next(iter(valLoader))

    #checking to make sure splits are correct
    print(len(trainLoader))
    print(len(valLoader))

    # --------------
    # Task 3
    # --------------
    model = createModel()

    #checking to make sure model was made correctly
    print(model)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"TotalParams: {total}")
    print(f"TrainableParams: {trainable}")


main()