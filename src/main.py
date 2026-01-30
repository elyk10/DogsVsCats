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
VAL_SPLIT = 0.1 # percentage for validation split size
BATCH_SIZE = 64 # number of data samples being used in each batch
WORKERS = 0 # number of parallel processes for loading data in
EPOCHS = 5

SEED = 18
gen = Generator()
gen.manual_seed(SEED)

device = device("cuda" if torch.cuda.is_available() else "cpu") # will use gpu if available

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
    model.to(device)

    return model

def trainModel(model, loader, optimizer, criterion):
    model.train()

    runningLoss = 0 # running total of how wrong the model prediction is and how confident it is in it
    runningCorrects = 0 # running total of how correct the model is

    for inputs, labels in loader:
        optimizer.zero_grad() # zero gradient for every batch
        outputs = model(inputs) # making predictions
        loss = criterion(outputs, labels) # calculate loss
        loss.backward()
        optimizer.step() # adjust weights

        runningLoss += loss.item() * inputs.size(0) # average loss of outputs multiplied by batch size to later average runningLoss by dataset size
        runningCorrects += (outputs.argmax(1) == labels).sum().item() # adds the amount of correct predictions to the running total


    return runningLoss / len(loader.dataset), runningCorrects / len(loader.dataset)




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
    valSize = int(len(dataset) * VAL_SPLIT)
    testSize = len(dataset) - trainSize - valSize

    ##TODO implement test split as well --------------------------------------------
    trainDataset, valDataset, testDataset = random_split(dataset, [trainSize, valSize, testSize], gen)
    

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

    # --------------
    # Task 4
    # --------------
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr = 0.001,         # how big a stem the optimizer takes
                                momentum = 0.9)     # how much of precious update is rememebered
    criterion = nn.CrossEntropyLoss() # logs probability of correct class

    for epoch in range(EPOCHS):
        #trainLoss, trainAcc = trainModel(model, trainLoader, optimizer, criterion)
        print(f"EPOCH {epoch}:")


main()