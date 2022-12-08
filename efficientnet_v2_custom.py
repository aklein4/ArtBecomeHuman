
import torchvision
import torch.nn as nn
import pytorch_lightning as pl
import torch


class EFFICIENTNET_V2_CUSTOM(pl.LightningModule):
    def __init__(self, lr=1e-5, n_classes=2, pretrained=False, grayscale=False):
        super().__init__()

        # make learning rate an arg
        self.lr = lr

        # number of classes
        self.n_classes = n_classes

        # yoink effnet
        self.effnet = torchvision.models.efficientnet_v2_s(weights=('IMAGENET1K_V1' if pretrained else None))
        
        # get correct output size
        self.effnet.classifier[-1] = nn.Linear(self.effnet.classifier[-1].in_features, self.n_classes)

        # get correct input size
        self.grayscale = grayscale
        if self.grayscale:

            old_weights = self.effnet.features[0][0].weight.data[:, 0, :, :].clone()
            old_weights = torch.unsqueeze(old_weights, 1)

            self.effnet.features[0][0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            
            conv = self.effnet.features[0][0]
            conv.weight.requires_grad = False
            conv.weight.data = old_weights
            conv.weight.requires_grad = True

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
        x, y = batch

        pred = self(x)

        loss = self.loss_func(pred, y)
        self.log('train_loss', loss, on_epoch=True, on_step=False, prog_bar=True)

        fixed = torch.maximum(torch.sign(pred[:, 1] - pred[:, 0]), torch.tensor(0, dtype=pred.dtype, device=pred.device))

        acc = torch.sum(y * fixed + (1-y) * (1-fixed)) / torch.numel(y)
        self.log('train_acc', acc, on_epoch=True, on_step=False, prog_bar=True)

        return loss


    def validation_step(self, batch, batch_idx):
        # get validation loss
        x, y = batch

        pred = self(x)

        loss = self.loss_func(pred, y)
        self.log('valid_loss', loss, on_epoch=True, on_step=False, prog_bar=True)

        fixed = torch.maximum(torch.sign(pred[:, 1] - pred[:, 0]), torch.tensor(0, dtype=pred.dtype, device=pred.device))

        acc = torch.sum(y * fixed + (1-y) * (1-fixed)) / torch.numel(y)
        self.log('val_acc', acc, on_epoch=True, on_step=False, prog_bar=True)

        return loss

