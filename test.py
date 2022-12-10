
import torch
import pytorch_lightning as pl

import argparse
import sys
import os
import random

from efficientnet_v2_custom import EFFICIENTNET_V2_CUSTOM
from binarydataset import BinaryDataset

REAL_PATH = "C:/Repos/data/testing"
AI_PATH = "jbkjbh"

def main(args):

    # load training data
    print("\nloading test data...")
    val_data = BinaryDataset(REAL_PATH, AI_PATH, skip_len=1, grayscale=False, noise=0.05)
    print("Test Data Sizes: Real -", len(val_data) - torch.sum(val_data.labels).item(), "AI -", torch.sum(val_data.labels).item())

    # load model checkpoint from training
    checkpoint = torch.load(args.path, map_location=args.device)

    # create model
    model = EFFICIENTNET_V2_CUSTOM(grayscale=False, legacy=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(args.device)
    model.eval()

    msg = ""
    correct = 0
    # loop through entire dataset
    inds = list(range(len(val_data)))
    random.shuffle(inds)
    for t in range(len(val_data)):
        i = inds[t]
        x, y = val_data[i]

        # get model's prediction (prediction is index of highest output)
        pred = model.forward(torch.unsqueeze(x, 0).to(args.device))
        guess = 1 if pred[0][1] > pred[0][0] else 0

        # print("<", pred[0][0].item(), pred[0][1].item(), ">", y.item())

        # check if correct
        if guess == y:
            correct += 1

        # progress message
        if t % 10 == 0:
            for _ in range(len(msg)):
                sys.stdout.write('\b')
            for _ in range(len(msg)):
                sys.stdout.write(' ')
            for _ in range(len(msg)):
                sys.stdout.write('\b')
            msg = str(t)+"/"+str(len(val_data)) + "  -- " + str(round(correct/(t+1), 3))
            sys.stdout.write(msg)
            sys.stdout.flush()
        
    print("\nValidaion set accuracy:", correct/len(val_data), "("+str(correct)+"/"+str(len(val_data))+")")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model to identify AI art.')

    parser.add_argument('--gpu', dest='device', action='store_const', const=torch.device('cuda'), default=torch.device('cpu'), 
                    help='Whether to use cuda gpu acceleration')

    parser.add_argument('-p', '--path', dest='path', type=str, default=None, 
                    help='Whether to use cuda gpu acceleration')

    args = parser.parse_args()
    if args.path is None:
        raise ValueError("Please input path to checkpoint.")
    main(args)
