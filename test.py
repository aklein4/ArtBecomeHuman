
import torch
import pytorch_lightning as pl

import argparse
import sys
import os

from efficientnet_v2_custom import EFFICIENTNET_V2_CUSTOM
from binarydataset import BinaryDataset

REAL_PATH = "./artbench"
AI_PATH = "./__AI__artbench"

def main(args):

    # load training data
    print("\nloading validation data...")
    val_data = BinaryDataset("./checkpoints", "./_ai_validation", skip_len=1)
    print("Training Data Sizes: Real -", torch.sum(val_data.labels[:, 0]).item(), "AI -", torch.sum(val_data.labels[:, 1]).item())

    # load model checkpoint from training
    checkpoint = torch.load("./checkpoints/epoch=0-step=782.ckpt")

    # create model
    model = EFFICIENTNET_V2_CUSTOM()
    model.load_state_dict(checkpoint['state_dict'])
    model.to(args.device)
    model.eval()

    msg = ""
    correct = 0
    # loop through entire dataset
    for i in range(len(val_data)):
        point = val_data[i]

        # get model's prediction (prediction is index of highest output)
        pred = model.forward(torch.unsqueeze(point['x'], 0).to(args.device))
        guess = 0
        if pred[0][1] > pred[0][0]:
            guess = 1

        # check if correct
        if guess == point['y'][1]:
            correct += 1

        # progress message
        if i % 10 == 0:
            for _ in range(len(msg)):
                sys.stdout.write('\b')
            msg = str(i)+"/"+str(len(val_data)) + "  -- " + str(round(correct/(i+1), 3))
            sys.stdout.write(msg)
            sys.stdout.flush()
        
    print("\nValidaion set accuracy:", correct/len(val_data), "("+str(correct)+"/"+str(len(val_data))+")")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model to identify AI art.')

    parser.add_argument('--gpu', dest='device', action='store_const', const=torch.device('cuda'), default=torch.device('cpu'), 
                    help='Whether to use cuda gpu acceleration')

    args = parser.parse_args()
    main(args)
