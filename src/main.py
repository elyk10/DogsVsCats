from dataset import ImageDataset
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np



def main():
    
    root = Path(__file__).parent
    datasetDir = root / ".." / "data"
    dataset = ImageDataset(datasetDir)
    print(hasattr(dataset, "__getitem__"))
    print(type(dataset))
    print(f"Length of dataset: {len(dataset)}")
    image, label = dataset[3]
    print(type(image))
    print(label)

    imgArray = np.asarray(image)
    plt.imshow(imgArray)
    plt.show()
    

main()