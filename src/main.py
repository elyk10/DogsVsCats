from dataset import ImageDataset
from pathlib import Path # for file paths 
import matplotlib.pyplot as plt # for graph plotting and displaying
import numpy as np # for array manipulation

from torch import Generator
from torch.utils.data import random_split, DataLoader # random_split - to split dataset into train and val | Dataloader - to create batches of dataset
from torchvision import transforms

#constants
ROOT = Path(__file__).parent / ".."

IMG_SIZE = 200 # height and width to change image to
TRAIN_SPLIT = 0.8 # percentage to split the data by
BATCH_SIZE = 64 # number of data samples being used in each batch
WORKERS = 0 # number of parallel processes for loading data in

SEED = 18
gen = Generator()
gen.manual_seed(SEED)

# obj of transform used to transform images when loaded in with given size, to a tensor and given normals
means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]
transformOBJ = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), transforms.Normalize(means, stds)])

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

    batchOne = next(iter(trainLoader))
    batchTwo = next(iter(valLoader))

    print(len(trainLoader))
    print(len(valLoader))

main()