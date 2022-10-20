
import torch
import numpy as np
import pytorch_lightning as pl
import torchvision

import EFFICIENTNET_V2
import os
from PIL import Image
import matplotlib.pyplot as plt

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, real_path, ai_path):
        to_image = torchvision.transforms.ToTensor()
        resize = torchvision.transforms.Resize([256, 256])

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

def main():

    dataset = MyDataset("./training_images/nebulas/", "./training_images/nebulas/")
    train_loader = torch.utils.data.DataLoader(dataset)

    model = EFFICIENTNET_V2.EFFICIENTNET_V2_CUSTOM()
    optimizer = model.configure_optimizers()

    trainer = pl.Trainer()
    trainer.fit(model, train_loader)

if __name__ == '__main__':
    main()