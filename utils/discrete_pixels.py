
import matplotlib.pyplot as plt
import argparse
import os
import random
from PIL import Image

import torch 
import torchvision
import numpy as np

IMAGE_SIZE = 256

GRAY = False

def main(args):
    blur = torchvision.transforms.GaussianBlur(5, sigma=1)


    imgs = os.listdir(args.path)
    random.shuffle(imgs)

    to_tensor = torchvision.transforms.PILToTensor()
    resize = torchvision.transforms.Resize([IMAGE_SIZE, IMAGE_SIZE])
    to_gray = torchvision.transforms.Grayscale(num_output_channels=1)

    for img in imgs:

        og = (resize(to_tensor(Image.open(args.path + "/" + img)))).to(torch.float32) / 255.0
        if GRAY:
            og = to_gray(og)
            og = torch.stack([og[0, :, :] for _ in range(3)])

        rounded = og.clone()
        rounded = torch.round(og * (args.bins-1)) / (args.bins-1)
        if GRAY:
            noise = torch.randn(og.shape[1:]) * 0.05
            rounded += torch.stack([noise for _ in range(3)])
        # else:
        #     rounded += torch.randn(og.shape) * 0.05
        # rounded = torch.maximum(rounded, torch.tensor(0))
        # rounded = torch.minimum(rounded, torch.tensor(1))
        # rounded = blur(rounded)

        og_picture = np.asarray(torch.permute(og, (1, 2, 0)))
        
        rounded_picture = np.asarray(torch.permute(rounded, (1, 2, 0)))

        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(6, 3)

        ax[0].imshow(og_picture, interpolation='nearest')
        ax[1].imshow(rounded_picture, interpolation='nearest')
        plt.savefig("rounded_out.png")
        input("continue...")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model to identify AI art.')

    parser.add_argument('-p', '--path', dest='path', type=str, default=None, 
                    help='Whether to use cuda gpu acceleration')
    
    parser.add_argument('-b', '--bins', dest='bins', type=int, default=10, 
                    help='Whether to use cuda gpu acceleration')

    args = parser.parse_args()
    if args.path is None:
        raise ValueError("Please input path to checkpoint.")
    main(args)
