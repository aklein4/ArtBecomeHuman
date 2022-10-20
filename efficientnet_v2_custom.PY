
import torchvision
import torch.nn as nn
import pytorch_lightning as pl
import torch

class EFFICIENTNET_V2_CUSTOM(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.effnet = torchvision.models.efficientnet_v2_s(weights='IMAGENET1K_V1')
        self.loss_func = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.parameters())
        return optimizer

    def forward(self, x):
        return self.effnet(x)

    def training_step(self, batch, batch_idx):
        pred = self(batch['x'])
        loss = self.loss_func(pred, batch['y'])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch['x'])
        loss = self.loss_func(pred, batch['y'])
        self.log('valid_loss', loss)
        return loss
