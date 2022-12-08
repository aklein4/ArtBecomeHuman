
import torch
import pytorch_lightning as pl

import argparse
import sys
import os

from efficientnet_v2_custom import EFFICIENTNET_V2_CUSTOM
from binarydataset import BinaryDataset

REAL_PATH = "./data/artbench"
AI_PATH = "./data/old_data"

def main(args):

    # load training data
    print("\nloading validation data...")
    val_data = BinaryDataset("./checkpoints", os.path.join(AI_PATH, "test/"), skip_len=20, grayscale=True)
    print("Training Data Sizes: Real -", len(val_data) - torch.sum(val_data.labels).item(), "AI -", torch.sum(val_data.labels).item())

    # load model checkpoint from training
    checkpoint = torch.load("./checkpoints/gray_checkpoints/last.ckpt", map_location=args.device)

    # create model
    model = EFFICIENTNET_V2_CUSTOM(grayscale=True)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(args.device)
    model.eval()

    msg = ""
    correct = 0
    # loop through entire dataset
    for i in range(len(val_data)):
        x, y = val_data[i]

        # get model's prediction (prediction is index of highest output)
        pred = model.forward(torch.unsqueeze(x, 0).to(args.device))
        guess = 0
        if pred[0][1] > pred[0][0]:
            guess = 1

        # check if correct
        if guess == y:
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
