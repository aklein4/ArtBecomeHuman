
import torch
import torchvision

import os
from PIL import Image

import numpy as np
import cv2
import os


IMAGE_SIZE = 256


def scrape_folder(folder) -> list:

    l = []
    for root, dirs, files in os.walk(folder):

        for f in files:
            if f[-4:] == ".jpg":
                l.append(os.path.join(root, f))

    return l


class BinaryDataset(torch.utils.data.Dataset):
    def __init__(self, real_path, ai_path, skip_len=1):
        # define transform classes
        to_tensor = torchvision.transforms.PILToTensor()
        resize = torchvision.transforms.Resize([IMAGE_SIZE, IMAGE_SIZE])

        real_files = scrape_folder(real_path)[::skip_len]
        ai_files = scrape_folder(ai_path)[::skip_len]
        combined = ai_files + real_files

        self.data = torch.zeros((len(combined), 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.uint8)
        self.labels = torch.zeros((len(combined), 2))

        print("Dataset Size:", round((self.data.element_size() * self.data.nelement())*1e-9, 4), "GB")

        tick = np.ceil(len(combined)/100)

        f_num = -1
        for f in combined:
            f_num += 1
            if f_num % tick == 0:
                print(f_num//tick, '%')
            self.data[f_num] = (resize(to_tensor(Image.open(f))))

        for i in range(len(ai_files)):
            self.labels[i][1] = 1
        for i in range(len(real_files)):
            self.labels[-i][0] = 1

        self.len = len(combined)
        

    def __len__(self):
        # total length of data
        return self.len
    
    def __getitem__(self,item):
        # check bounds
        if item >= self.len or item < 0:
            raise ValueError("Dataloader index out of range.")
        
        # higher is ai
        return {
            'x': self.data[item].to(torch.float32) / 255.0,
            'y': self.labels[item].to(torch.float32)
        }
    

    def to(self, device):
        # move all data to device
        for elem in self.real_imgs:
            elem.to(device)
        for elem in self.ai_imgs:
            elem.to(device)
    
    def get_img(self, item):
        return np.asarray(torch.permute(self[item]['x'], (1, 2, 0)))