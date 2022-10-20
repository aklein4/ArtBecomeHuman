
import torch

import os
import sys
sys.path.append(os.getcwd()+"/models")
import EFFICIENTNET_V2

def main():

    

    dataloader = torch.utils.data.Dataloader()

    model = EFFICIENTNET_V2.EFFICIENTNET_V2_CUSTOM()

if __name__ == '__main__':
    main()