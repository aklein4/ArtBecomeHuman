
import torchvision
import torch.nn as nn
import pytorch_lightning as pl
import torch

class EFFICIENTNET_V2_CUSTOM(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()

        # make learning rate an argument
        self.lr = lr

        # yoink effnet
        self.effnet = torchvision.models.efficientnet_v2_s(weights='IMAGENET1K_V1')
        
        # use this as loss
        self.loss_func = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        # create optimizer with our lr
        optimizer = torch.optim.AdamW(params=self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x):
        # pass x through net
        return self.effnet(x)[:,:2]

    def training_step(self, batch, batch_idx):
        # get training loss
        pred = self(batch['x'])
        loss = self.loss_func(pred, batch['y'])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # get validation loss
        pred = self(batch['x'])
        loss = self.loss_func(pred, batch['y'])
        self.log('valid_loss', loss)
        return loss
