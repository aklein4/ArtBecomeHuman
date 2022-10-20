
import torch
import torchvision

import os
from PIL import Image

IMAGE_SIZE = 256

class BinaryDataset(torch.utils.data.Dataset):
    def __init__(self, real_path, ai_path, max_size=10000):
        to_image = torchvision.transforms.ToTensor()
        resize = torchvision.transforms.Resize([IMAGE_SIZE, IMAGE_SIZE])

        self.real_imgs = []
        for f in os.listdir(real_path):
            img = resize(to_image(Image.open(real_path+f)))
            if img.shape[0] == 3:
                self.real_imgs.append(img)
            if len(self.real_imgs) >=  max_size:
                break

        self.ai_imgs = []
        for f in os.listdir(ai_path):
            img = resize(to_image(Image.open(ai_path+f)))
            if img.shape[0] == 3:
                self.ai_imgs.append(img)
            if len(self.ai_imgs) >=  max_size:
                break

    def __len__(self):
        return len(self.real_imgs)+len(self.ai_imgs)
    
    def __getitem__(self,item):
        if not item < len(self.real_imgs)+len(self.ai_imgs):
            raise ValueException("Dataloader index out of range.")
        if item < len(self.real_imgs):
            return {
                'x': self.real_imgs[item],
                'y': 0
            }
        return {
            'x': self.ai_imgs[item-len(self.real_imgs)],
            'y': 1
        }
    
    def to(self, device):
        for elem in self.real_imgs:
            elem.to(device)
        for elem in self.ai_imgs:
            elem.to(device)