import torch
import os

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from skimage.color import lab2rgb
from tqdm import tqdm

from model import Net
from colorize_data import ColorizeData


if __name__ == "__main__":
    path = "../TestImages/"
    test_paths = os.listdir(path)
    paths = [os.path.join(path + x) for x in test_paths]
    model_test = Net()
    model_test.load_state_dict(torch.load('saved_model.pth', map_location=torch.device('cpu')))
    model_test.eval()

    test_dataset = ColorizeData(paths=paths)
    test_dataloader = DataLoader(test_dataset, batch_size=4)
    _, ax = plt.subplots(4, 3, figsize=(256, 256))
    gray_test_data, color_test_data = next(iter(test_dataloader))
    test_outputs = model_test(gray_test_data)
    for i in tqdm(range(4)):
        ax[i, 0].imshow(np.squeeze(gray_test_data[i].cpu().detach().numpy().transpose(1,2,0)), cmap='gray')
        ax[i, 1].imshow(test_outputs[i].cpu().detach().numpy().transpose(1,2,0))
        ax[i, 2].imshow(color_test_data[i].cpu().detach().numpy().transpose(1,2,0))
    
    plt.savefig("../Results/RGB_ResNet/ColorizedImages/Comparison.jpg", bbox_inches='tight')