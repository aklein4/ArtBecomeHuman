
import torch

# pip install grad-cam
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import argparse
import random
import matplotlib.pyplot as plt

from efficientnet_v2_custom import EFFICIENTNET_V2_CUSTOM
from binarydataset import BinaryDataset

def main(args):

    # load validation data
    val_data = BinaryDataset("./_non_AI_validation/", "./__ai_validation/", max_size=100)
    print("Validation Data Sizes: Real -", len(val_data.real_imgs), "AI -", len(val_data.ai_imgs))

    # load model state
    state_dict = torch.load("./model_states/epoch=327-step=65928.pt", map_location=args.device)

    # create model
    model = EFFICIENTNET_V2_CUSTOM()
    model.to(args.device)
    model.load_state_dict(state_dict)
    model.eval()

    cam = GradCAM(model=model, target_layers=[model.effnet.features[7]], use_cuda=(True if args.device==torch.device("cuda") else False))

    while True:
        item = random.randrange(0, len(val_data))
        img = val_data.get_img(item)
        img_tensor = torch.unsqueeze(val_data[item]['x'], 0)
        ai = val_data[item]['y']
        targets = [ClassifierOutputTarget(ai)]

        print("AI?", "yes" if ai == 1 else "no")
        pred = model.forward(img_tensor.to(args.device))
        guess = 0
        if pred[0][1] > pred[0][0]:
            guess = 1
        print("Predicts AI?", "yes" if guess == 1 else "no")
        print(" ----- ")

        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)

        visualization = show_cam_on_image(img, grayscale_cam[0], use_rgb=True)
        plt.imshow(visualization)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model to identify AI art.')

    parser.add_argument('--gpu', dest='device', action='store_const', const=torch.device('cuda'), default=torch.device('cpu'), 
                    help='Whether to use cuda gpu acceleration')

    args = parser.parse_args()
    main(args)