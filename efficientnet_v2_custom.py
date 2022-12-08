
import torchvision
import torch.nn as nn
import pytorch_lightning as pl
import torch


class EFFICIENTNET_V2_CUSTOM(pl.LightningModule):
    def __init__(self, lr=1e-5, n_classes=2, pretrained=False):
        super().__init__()

        # make learning rate an arg
        self.lr = lr

        # number of classes
        self.n_classes = n_classes

        # yoink effnet
        self.effnet = torchvision.models.efficientnet_v2_s(weights=('IMAGENET1K_V1' if pretrained else None))
        
        # get correct output size
        self.effnet.classifier[-1] = nn.Linear(self.effnet.classifier[-1].in_features, self.n_classes)

        # use this as loss
        self.loss_func = nn.CrossEntropyLoss()


    def configure_optimizers(self):
        # create optimizer with our lr
        optimizer = torch.optim.AdamW(params=self.parameters(), lr=self.lr)
        return optimizer


    def forward(self, x):
        # pass x through net
        return self.effnet(x)


    def training_step(self, batch, batch_idx):
        # get training loss
        pred = self(batch['x'])

        loss = self.loss_func(pred, batch['y'])
        self.log('train_loss', loss, on_epoch=True, on_step=False, prog_bar=True)

        fixed = torch.maximum(torch.sign(pred[:, 1] - pred[:, 0]), torch.tensor(0, dtype=pred.dtype, device=pred.device))

        acc = torch.sum(batch['y'][:, 1] * fixed + (1-batch['y'][:, 1]) * (1-fixed)) / torch.numel(batch['y'][:, 1])
        self.log('train_acc', acc, on_epoch=True, on_step=False, prog_bar=True)

        return loss


    def validation_step(self, batch, batch_idx):
        # get validation loss
        pred = self(batch['x'])

        loss = self.loss_func(pred, batch['y'])
        self.log('valid_loss', loss, on_epoch=True, on_step=False, prog_bar=True)

        fixed = torch.maximum(torch.sign(pred[:, 1] - pred[:, 0]), torch.tensor(0, dtype=pred.dtype, device=pred.device))

        acc = torch.sum(batch['y'][:, 1] * fixed + (1-batch['y'][:, 1]) * (1-fixed)) / torch.numel(batch['y'][:, 1])
        self.log('val_acc', acc, on_epoch=True, on_step=False, prog_bar=True)

        return loss
