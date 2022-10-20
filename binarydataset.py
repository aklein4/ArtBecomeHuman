
import torch
import torchvision

import os
from PIL import Image

import numpy as np
import cv2

IMAGE_SIZE = 256

class BinaryDataset(torch.utils.data.Dataset):
    def __init__(self, real_path, ai_path, max_size=10000):
        # define transform classes
        self.to_image = torchvision.transforms.ToTensor()
        self.resize = torchvision.transforms.Resize([IMAGE_SIZE, IMAGE_SIZE])

        # # load non-ai images
        self.real_imgs = []
        for f in os.listdir(real_path):
            img = self.to_image(Image.open(real_path+f))
            if img.shape[0] == 3:
                self.real_imgs.append(real_path+f)

        # load ai images
        self.ai_imgs = []
        for f in os.listdir(ai_path):
            img = self.to_image(Image.open(ai_path+f))
            if img.shape[0] == 3:
                self.ai_imgs.append(ai_path+f)

        # # load non-ai images
        # self.real_imgs = []
        # for f in os.listdir(real_path):
        #     img = resize(to_image(Image.open(real_path+f)))
        #     if img.shape[0] == 3:
        #         self.real_imgs.append(img)
        #     if len(self.real_imgs) >=  max_size:
        #         break

        # load ai images
        # self.ai_imgs = []
        # for f in os.listdir(ai_path):
        #     img = resize(to_image(Image.open(ai_path+f)))
        #     if img.shape[0] == 3:
        #         self.ai_imgs.append(img)
        #     if len(self.ai_imgs) >=  max_size:
        #         break

    def __len__(self):
        # total length of data
        return len(self.real_imgs)+len(self.ai_imgs)
    
    def __getitem__(self,item):
        # check bounds
        if not item < len(self.real_imgs)+len(self.ai_imgs):
            raise ValueError("Dataloader index out of range.")
        
        # lower index is real
        if item < len(self.real_imgs):
            return {
                'x': self.resize(self.to_image(Image.open(self.real_imgs[item]))),
                'y': 0
            }
        # higher is ai
        return {
            'x': self.resize(self.to_image(Image.open(self.ai_imgs[item-len(self.real_imgs)]))),
            'y': 1
        }

        # # lower index is real
        # if item < len(self.real_imgs):
        #     return {
        #         'x': self.real_imgs[item],
        #         'y': 0
        #     }
        # # higher is ai
        # return {
        #     'x': self.ai_imgs[item-len(self.real_imgs)],
        #     'y': 1
        # }
    
    def to(self, device):
        # move all data to device
        for elem in self.real_imgs:
            elem.to(device)
        for elem in self.ai_imgs:
            elem.to(device)
    
    def get_img(self, item):
        # check bounds
        if not item < len(self.real_imgs)+len(self.ai_imgs):
            raise ValueError("Dataloader index out of range.")
        
        img = None
        # lower index is real
        if item < len(self.real_imgs):
            img = np.asarray(Image.open(self.real_imgs[item]))
        # higher is ai
        else:
            img = np.asarray(Image.open(self.ai_imgs[item-len(self.real_imgs)]))
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = np.float32(img) / 255
        return img