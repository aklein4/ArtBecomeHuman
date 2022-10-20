
import torch
import torchvision

import os
from PIL import Image

IMAGE_SIZE = 256
MAX_SET_SIZE = 5

class BinaryDataset(torch.utils.data.Dataset):
    def __init__(self, real_path, ai_path):
        to_image = torchvision.transforms.ToTensor()
        resize = torchvision.transforms.Resize([IMAGE_SIZE, IMAGE_SIZE])

        self.real_imgs = []
        for f in os.listdir(real_path):
            img = resize(to_image(Image.open(real_path+f)))
            if img.shape[0] == 3:
                self.real_imgs.append(img)
            if len(self.real_imgs) >=  MAX_SET_SIZE:
                break

        self.ai_imgs = []
        for f in os.listdir(ai_path):
            img = resize(to_image(Image.open(ai_path+f)))
            if img.shape[0] == 3:
                self.ai_imgs.append(img)
            if len(self.ai_imgs) >=  MAX_SET_SIZE:
                break

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