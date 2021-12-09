import torchvision.transforms as T

from torch.utils.data import Dataset
from torchvision.transforms.functional import resize
from skimage.color import rgb2lab
from PIL import Image


class ColorizeData(Dataset):
    def __init__(self, paths):
        # Initialize dataset, you may use a second dataset for validation if required
        # Use the input transform to convert images to grayscale
        self.input_transform = T.Compose([T.ToTensor(),
                                          T.Resize(size=(256,256)),
                                          T.Grayscale(),
                                          T.Normalize((0.5), (0.5))
                                          ])
        # Use this on target images(colorful ones)
        self.target_transform = T.Compose([T.ToTensor(),
                                           T.Resize(size=(256,256)),
                                           T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])    
        self.paths = paths

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        # Return the input tensor and output tensor for training
        image = Image.open(self.paths[index]).convert("RGB")
        input_image = self.input_transform(image)
        target_image = self.target_transform(image)
        return (input_image, target_image)  