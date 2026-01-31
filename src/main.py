from dataset import ImageDataset
from pathlib import Path # for file paths 
import matplotlib.pyplot as plt # for graph plotting and displaying
import numpy as np # for array manipulation
from tqdm import tqdm # used for progress bar while completing epochs
import csv
import shutil
from multiprocessing import freeze_support

from torch import Generator, device
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, Subset # random_split - to split dataset into train and val | Dataloader - to create batches of dataset | Subset - to use subset of data for trail
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

#constants
#file paths
ROOT = Path(__file__).parent / ".."
OUTPUT_PATH = Path("output") 
MODEL_DIR = OUTPUT_PATH / "models"
MODEL_DIR.mkdir(parents = True, exist_ok = True)
PRED_DIR = OUTPUT_PATH / "predictions"
PRED_DIR.mkdir(parents = True, exist_ok = True)

IMG_SIZE = 256 # height and width to change image to
CROP_SIZE = 224
TRAIN_SPLIT = 0.72 # percentage to split the data by
VAL_SPLIT = 0.18 # percentage for validation split size
BATCH_SIZE = 64 # number of data samples being used in each batch
WORKERS = 2 # number of parallel processes for loading data in
EPOCHS = 5

SEED = 18
gen = Generator()
gen.manual_seed(SEED)

device = device("cuda" if torch.cuda.is_available() else "cpu") # will use gpu if available

# obj of transform used to transform images when loaded in with given size, to a tensor and given normals
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transformOBJ = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                  # transforms 
                                   transforms.CenterCrop(CROP_SIZE), 
                                   transforms.ToTensor(), 
                                   transforms.Normalize(mean, std)])

def collateSkip(batch): # method created to skip data entries in batch that are none
    batch = [i for i in batch if i is not None]
    if len(batch == 0):
        return None
    return torch.utils.data.default_collate(batch)

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

    for inputs, labels, _ in tqdm(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad() # zero gradient for every batch
        outputs = model(inputs) # making predictions
        loss = criterion(outputs, labels) # calculate loss
        loss.backward()
        optimizer.step() # adjust weights

        runningLoss += loss.item() * inputs.size(0) # average loss of outputs multiplied by batch size to later average runningLoss by dataset size
        runningCorrects += (outputs.argmax(1) == labels).sum().item() # adds the amount of correct predictions to the running total
        
    return runningLoss / len(loader.dataset), runningCorrects / len(loader.dataset)

def valModel(model, loader, criterion): # same as training model just no self correcting
    model.eval()

    runningLoss = 0 # running total of how wrong the model prediction is and how confident it is in it
    runningCorrects = 0 # running total of how correct the model is
    
    with torch.no_grad():
        for inputs, labels, _ in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs) # making predictions
            loss = criterion(outputs, labels) # calculate loss
        
            runningLoss += loss.item() * inputs.size(0) # average loss of outputs multiplied by batch size to later average runningLoss by dataset size
            runningCorrects += (outputs.argmax(1) == labels).sum().item() # adds the amount of correct predictions to the running total


    return runningLoss / len(loader.dataset), runningCorrects / len(loader.dataset)

def predModel(model, loader):
    model.eval()
    results = []

    with torch.no_grad():
        for inputs, folders, fileNames in tqdm(loader, desc = "Predicting"):
            inputs = inputs.to(device) 
            outputs = model(inputs)
            # get probability of output being all classes
            probs = torch.softmax(outputs, dim = 1)
            preds = probs.argmax(1).cpu().numpy()
            # get confidence in prediction
            confs = probs.max(1).values.cpu().numpy()

            # go through files and create tuples stored in results with file path and label of classification
            for fName, folder, pred, conf in zip(fileNames, folders, preds, confs):
                # prediction label
                label = "cat"
                if pred == 1:
                    label = "dog"
                # folder image was in (cat or dog) to identify which image
                folderName = "Cat"
                if folder == 1:
                    folderName = "Dog"
                results.append((folderName + fName, label, conf))
    
    return results
                


def main():
    print("Computer Vision Project Initialized")
    # --------------
    # Task 1
    # --------------
    datasetDir = ROOT / "data"
    dataset = ImageDataset(datasetDir, transformOBJ)

    ### -- for test purposes
    #subsetSize = int(len(dataset) * 0.01)
    #torch.manual_seed(SEED)
    #indices = torch.randperm(len(dataset))[:subsetSize]
    #dataset = Subset(dataset, indices)

    print(f"Length of dataset: {len(dataset)}")
    image, label, _ = dataset[3]
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

    trainDataset, valDataset, testDataset = random_split(dataset, [trainSize, valSize, testSize], gen)
    

    trainLoader = DataLoader(trainDataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = WORKERS, collate_fn = collateSkip)
    valLoader = DataLoader(valDataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = WORKERS, collate_fn = collateSkip)
    testLoader = DataLoader(testDataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = WORKERS, collate_fn = collateSkip)

    #checking to make sure loaders can be iterated
    #batchOne = next(iter(trainLoader))
    #batchTwo = next(iter(valLoader))

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
                                lr = 0.001,         # how big a step the optimizer takes
                                momentum = 0.9)     # how much of precious update is rememebered
    criterion = nn.CrossEntropyLoss() # logs probability of correct class

    tLoss = []
    tAcc = []
    vLoss = []
    vAcc = []
    # iterate through number of epochs
    for epoch in range(EPOCHS):
        print(f"EPOCH [{epoch + 1} / {EPOCHS}]:")
        trainLoss, trainAcc = trainModel(model, trainLoader, optimizer, criterion)
        valLoss, valAcc = valModel(model, valLoader, criterion)

        print(f"Train Loss: {trainLoss}, Train Accuraccy: {trainAcc}")
        tLoss.append(trainLoss)
        tAcc.append(trainAcc)
        print(f"Validation Loss: {valLoss}, Validation Accuracy: {valAcc}")
        vLoss.append(valLoss)
        vAcc.append(valAcc)

    plt.plot(tLoss, label = "Training Loss")
    plt.plot(vLoss, label = "Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.show()

    plt.plot(tAcc, label = "Training Accuracy")
    plt.plot(vAcc, label = "Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.show()
        
    # --------------
    # Task 6
    # --------------
    
    # save model to given model directory
    modelPath = MODEL_DIR / "modelCatsVsDogs.pth"
    torch.save(model.state_dict(), modelPath)
    print(f"Model saved to {MODEL_DIR}")

    # load saved model
    savedModel = resnet18()
    inFeatures = model.fc.in_features
    savedModel.fc = nn.Linear(inFeatures, 2)
    savedModel.load_state_dict(torch.load(modelPath, weights_only = True))
    print(f"Model {modelPath} loaded in from disk")

    # make predictions with saved model and save in a csv or txt file
    # save a few predicted images with label as well
    print("Making predictions on test dataset")
    preds = predModel(savedModel, testLoader)
    csvPath = PRED_DIR / "predictions.csv"
    # write predictions into csv file
    with open(csvPath, "w", newline = "") as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "predicted_label", "probability"])
        writer.writerows(preds)

    # take select images from predictions and move them into prediction file, renaming with prediction
    predPhotos = 10 if len(preds) > 10 else len(preds)
    i = 0
    for pred in preds:
        i += 1
        if i >= predPhotos: # only does 10 photos or max length of preds
            break
        
        fileAndFolderName = pred[0]
        fileName = fileAndFolderName[3:]
        folderName = fileAndFolderName[:3]
        src = datasetDir / folderName / fileName
        outFile = "pred" + str(i) + pred[1] + ".jpg"
        dst = PRED_DIR / outFile
        
        shutil.copy2(src, dst)



    print(f"Predictions saved to {PRED_DIR}")

if __name__ == "__main__":
    freeze_support()
    main()