
import torch

# pip install grad-cam
import pytorch_grad_cam
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import argparse
import random
import matplotlib.pyplot as plt
import os
from pynput import keyboard
import sys

from efficientnet_v2_custom import EFFICIENTNET_V2_CUSTOM
from binarydataset import BinaryDataset


REAL_PATH = "./artbench"
AI_PATH = "./__AI__artbench"

CAM_TYPE = pytorch_grad_cam.EigenCAM
    

def main(args):

    # load training data
    print("\nloading validation data...")
    val_data = BinaryDataset(os.path.join(REAL_PATH, "test/"), os.path.join(AI_PATH, "test/"), skip_len=100)
    print("Training Data Sizes: Real -", torch.sum(val_data.labels[:, 0]).item(), "AI -", torch.sum(val_data.labels[:, 1]).item())

    # load model checkpoint from training
    checkpoint = torch.load("./checkpoints/best_checkpoint.ckpt")

    # create model
    model = EFFICIENTNET_V2_CUSTOM()
    model.load_state_dict(checkpoint['state_dict'])
    model.to(args.device)
    model.eval()

    label = 1
    layer = len(model.effnet.features)-1

    while True:
        
        print("\n ----- \n")

        item = random.randrange(0, len(val_data))

        img_tensor = torch.unsqueeze(val_data[item]['x'], 0)

        print("AI?", "yes" if val_data[item]['y'][1] == 1 else "no")

        pred = model.forward(img_tensor.to(args.device))
        guess = 0
        if pred[0][1] > pred[0][0]:
            guess = 1
        print("Predicts AI?", "yes" if guess == 1 else "no")
        print("")

        act_ind = len(model.effnet.features[layer])-1

        msg = ""
        while True:
            
            # init cam
            cam = CAM_TYPE(model=model, target_layers=[model.effnet.features[layer][act_ind]], use_cuda=(True if args.device==torch.device("cuda") else False))

            img = val_data.get_img(item)
            img_tensor = torch.unsqueeze(val_data[item]['x'], 0)

            targets = [ClassifierOutputTarget(label)]

            grayscale_cam = cam(input_tensor=img_tensor, targets=targets)

            visualization = show_cam_on_image(img, grayscale_cam[0], use_rgb=True)
            plt.imshow(visualization)
            plt.savefig("vis_out.png")

            eraser = ""
            for _ in range(len(msg)):
                eraser += '\b'
            for _ in range(len(msg)):
                eraser += ' '
            for _ in range(len(msg)):
                eraser += '\b'
            msg = "Layer: " + str(layer) + ", Index: " + str(act_ind) + ", Type: " + model.effnet.features[layer][act_ind].__class__.__name__ + ", Label:" + ("AI" if label == 1 else "REAL") + " "
            sys.stdout.write(eraser + msg)
            sys.stdout.flush()

            command = None
            while True:
                with keyboard.Events() as events:
                    # Block for as much as possible
                    event = events.get(1e6)
                    try:
                        command = event.key.char
                        break
                    except:
                        if event.key == keyboard.Key.space:
                            command = ' '
                            break
                        pass

            if command == 'a':
                label = 0
            elif command == 'd':
                label = 1

            elif command == 'w':
                layer = min(len(model.effnet.features)-1, layer + 1)
                act_ind = len(model.effnet.features[layer])-1
            elif command == 's':
                layer = max(0, layer - 1)
                act_ind = len(model.effnet.features[layer])-1
            
            elif command == 'e':
                act_ind = min(len(model.effnet.features[layer])-1, act_ind + 1)
            elif command == 'q':
                act_ind = max(0, act_ind - 1)

            elif command == ' ':
                break
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model to identify AI art.')

    parser.add_argument('--gpu', dest='device', action='store_const', const=torch.device('cuda'), default=torch.device('cpu'), 
                    help='Whether to use cuda gpu acceleration')

    args = parser.parse_args()
    main(args)