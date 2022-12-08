
import torch
import torchvision

import os
from PIL import Image

import numpy as np
import os
from tqdm import tqdm


# Size of images that we are working with
IMAGE_SIZE = 256


def scrape_folder(folder) -> list:
    # get a list of the jpg files in this folder and sub-directories

    l = []
    for root, dirs, files in os.walk(folder):

        for f in files:
            if f[-4:] == ".jpg":
                l.append(os.path.join(root, f))

    return l


class BinaryDataset(torch.utils.data.Dataset):
    def __init__(self, real_path, ai_path, skip_len=1, verbose=True, grayscale=False):
        # define transform classes
        to_tensor = torchvision.transforms.PILToTensor()
        resize = torchvision.transforms.Resize([IMAGE_SIZE, IMAGE_SIZE])
        
        # handle grayscale
        self.grayscale = grayscale
        to_gray = torchvision.transforms.Grayscale(num_output_channels=1)

        # get list of real and ai files
        real_files = scrape_folder(real_path)[::skip_len]
        ai_files = scrape_folder(ai_path)[::skip_len]
        combined = ai_files + real_files

        # init tensors to fill with data
        # note that we use uint8 and convert to float32 later, this takes RAM usage from 80GB to 20GB
        self.data = torch.zeros((len(combined), (1 if self.grayscale else 3), IMAGE_SIZE, IMAGE_SIZE), dtype=torch.uint8)
        self.labels = torch.zeros(len(combined), dtype=torch.long)

        if verbose:
            print("Dataset Size:", round((self.data.element_size() * self.data.nelement())*1e-9, 4), "GB")

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
                if self.grayscale:
                    img = to_gray(img)

                # fill in data
                self.data[i] = img

        # ai corresponds to index 1, and it is first. human is left as 0
        for i in range(len(ai_files)):
            self.labels[i] = 1

        # save length
        self.len = len(combined)
        

    def __len__(self):
        # total length of data
        return self.len
    

    def __getitem__(self,item):
        # check bounds
        if item >= self.len or item < 0:
            raise ValueError("Dataloader index out of range.")

        # return x, y tuple
        return self.data[item].to(torch.float32) / 255.0, self.labels[item]

    
    def get_img(self, item):
        # return formatted as image
        x, y = self[item]
        return np.asarray(torch.permute(x, (1, 2, 0)))

