from dataset import ImageDataset
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).parent / ".."


def main():
    datasetDir = ROOT / "data"
    dataset = ImageDataset(datasetDir)
    print(f"Length of dataset: {len(dataset)}")
    image, label = dataset[3]
    print(type(image))
    print(label)

    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.title("Image 4 from dataset")
    plt.show()

    _, height, width = image.shape
    print(f"Image has width: {width}px and height: {height}px")
    

main()