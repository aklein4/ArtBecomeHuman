
import torchvision as tv
import torch.nn as nn
import torch

class EFFICIENTNET_V2_CUSTOM(nn.Module):
    def __init__(self):
        super().__init__()
        self.effnet = tv.models.efficientnet_v2_s(weights='IMAGENET1K_V1')

    def init_optimizer(self):
        self.optimizer = torch.optim.AdamW()

    def forward(self, x):
        return self.effnet(x)