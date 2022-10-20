
import torch
import pytorch_lightning as pl

import argparse
import sys

from efficientnet_v2_custom import EFFICIENTNET_V2_CUSTOM
from binarydataset import BinaryDataset

def main(args):

    # load training data
    train_data = BinaryDataset("./non_AI_art/", "./__ai_training/", max_size=900)
    print("Training Data Sizes: Real -", len(train_data.real_imgs), "AI -", len(train_data.ai_imgs))

    # load validation data
    val_data = BinaryDataset("./non_AI_validation/", "./__ai_validation/", max_size=100)
    print("Validation Data Sizes: Real -", len(val_data.real_imgs), "AI -", len(val_data.ai_imgs))

    # load model checkpoint from training
    checkpoint = torch.load("./checkpoints/epoch=327-step=65928.ckpt")

    # create model
    model = EFFICIENTNET_V2_CUSTOM()
    model.to(args.device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    msg = ""
    correct = 0
    # loop through entire dataset
    for i in range(len(train_data)):
        point = train_data[i]

        # get model's prediction (prediction is index of highest output)
        pred = model.forward(torch.unsqueeze(point['x'], 0).to(args.device))
        guess = 0
        if pred[0][1] > pred[0][0]:
            guess = 1
        
        # check if correct
        if guess == point['y']:
            correct += 1
        
        # progress message
        if i % 10 == 0:
            for _ in range(len(msg)):
                sys.stdout.write('\b')
            msg = str(i)+"/"+str(len(train_data))
            sys.stdout.write(msg)
            sys.stdout.flush()

    print("\nTraining set accuracy:", correct/len(train_data), "("+str(correct)+"/"+str(len(train_data))+")")

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
        if guess == point['y']:
            correct += 1

        # progress message
        if i % 10 == 0:
            for _ in range(len(msg)):
                sys.stdout.write('\b')
            msg = str(i)+"/"+str(len(train_data))
            sys.stdout.write(msg)
            sys.stdout.flush()
        
    print("\nValidaion set accuracy:", correct/len(val_data), "("+str(correct)+"/"+str(len(val_data))+")")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model to identify AI art.')

    parser.add_argument('--gpu', dest='device', action='store_const', const=torch.device('cuda'), default=torch.device('cpu'), 
                    help='Whether to use cuda gpu acceleration')

    args = parser.parse_args()
    main(args)
