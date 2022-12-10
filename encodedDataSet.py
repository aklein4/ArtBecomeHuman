
import torchvision
import torch.nn as nn
import pytorch_lightning as pl
import torch
from tqdm import tqdm
import numpy as np
from PIL import Image
import os

from binarydataset import scrape_folder


IMAGE_SIZE = 256


class Encoder(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # yoink effnet
        self.effnet = torchvision.models.efficientnet_v2_l(weights=('IMAGENET1K_V1'))
        self._encoding_size = self.effnet.classifier.in_features
        self.effnet.classifier = nn.Identity()

        self.eval()

    def forward(self, x):
        # pass x through net
        return self.effnet(x)

    def encoding_size(self):
        return self._encoding_size

    def encodeData(self, real_path, ai_path, save_folder, verbose=True):
        # define transform classes
        to_tensor = torchvision.transforms.PILToTensor()
        resize = torchvision.transforms.Resize([IMAGE_SIZE, IMAGE_SIZE])

        # get list of real and ai files
        real_files = scrape_folder(real_path)
        ai_files = scrape_folder(ai_path)
        combined = ai_files + real_files

        # check for empty (bad)
        if len(combined) == 0:
            raise RuntimeWarning("data is empty!")

        # init tensors to fill with data
        # note that we use uint8 and convert to float32 later, this takes RAM usage from 80GB to 20GB
        data = torch.zeros([len(combined), self._encoding_size], dtype=torch.float32)
        labels = torch.zeros(len(combined), dtype=torch.long)

        # print the size of the dataset
        if verbose:
            print("Human Images:", len(real_files), " -- AI Images:", len(ai_files))
            print("Dataset Size:", round((data.element_size() * data.nelement())*1e-9, 4), "GB")

        # tqdm seems to break on large numbers
        with (tqdm(range(1, 100)) if verbose else None) as bar:
            for i in range(len(combined)):
                if verbose:
                    old_n = bar.n
                    bar.n = np.floor(100*(i+1)/len(combined))
                    if bar.n != old_n:
                        bar.update()

                # get image from file
                img = (resize(to_tensor(Image.open(combined[i]))))
                ready = img.to(torch.float32) / 255.0

                encoding = self.forward(torch.unsqueeze(ready, 0))

                # fill in data
                data[i] = encoding

        # ai corresponds to index 1, and it is first. human is left as 0
        for i in range(len(ai_files)):
            labels[i] = 1

        os.makedirs(save_folder, exist_ok=True)
        torch.save(data, os.path.join(save_folder, "data.pt"))
        torch.save(labels, os.path.join(save_folder, "labels.pt"))


class EncodedDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, verbose=True):
        self.data = torch.load(os.path.join(data_folder, "data.pt"))
        self.labels = torch.load(os.path.join(data_folder, "labels.pt"))
    
        self.len = self.labels.shape[0]

        if verbose:
            print("Human Images:", self.len - torch.sum(self.labels).item(), " -- AI Images:", torch.sum(self.labels).item())
            print("Dataset Size:", round((self.data.element_size() * self.data.nelement())*1e-9, 4), "GB")


    def __len__(self):
        # total length of data
        return self.len
    

    def __getitem__(self,item):
        # check bounds
        if item >= self.len or item < 0:
            raise ValueError("Dataloader index out of range.")

        # return x, y tuple
        return  self.data[item], self.labels[item]