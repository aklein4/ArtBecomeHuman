
import torch
import torchvision

import os
from PIL import Image

IMAGE_SIZE = 256

class BinaryDataset(torch.utils.data.Dataset):
    def __init__(self, real_path, ai_path):
        to_image = torchvision.transforms.ToTensor()
        resize = torchvision.transforms.Resize([IMAGE_SIZE, IMAGE_SIZE])

        self.real_imgs = []
        for f in os.listdir(real_path):
            self.real_imgs.append(resize(to_image(Image.open(real_path+f))))

        self.ai_imgs = []
        for f in os.listdir(ai_path):
            self.real_imgs.append(resize(to_image(Image.open(real_path+f))))

    def __len__(self):
        return len(self.real_imgs)+len(self.ai_imgs)
    
    def __getitem__(self,item):
        if not item < len(self.real_imgs)+len(self.ai_imgs):
            raise ValueException("Dataloader index out of range.")
        if item < len(self.real_imgs):
            return {
                "x" : self.real_imgs[item],
                "y" : 0
            }
        return {
            "x" : self.ai_imgs[item-len(self.real_imgs)],
            "y" : 1
        }