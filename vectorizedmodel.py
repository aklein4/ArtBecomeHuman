
import torchvision
import torch.nn as nn
import pytorch_lightning as pl
import torch


class EFFICIENTNET_V2_CUSTOM(pl.LightningModule):
    def __init__(self, n_inputs, n_classes=2, lr=1e-5):
        super().__init__()

        self.n_inputs = n_inputs
        self.n_classes = n_classes
        self.lr = lr

        self.net = nn.Sequential(
            nn.Linear(self.n_inputs, 2048, bias=True),
            nn.ELU(inplace=True),
            nn.Dropout(p=0.25, inplace=True),

            nn.Linear(2048, 2048, bias=True),
            nn.ELU(inplace=True),
            nn.Dropout(p=0.25, inplace=True),

            nn.Linear(2048, 1024, bias=True),
            nn.ELU(inplace=True),
            nn.Dropout(p=0.25, inplace=True),

            nn.Linear(1024, self.n_classes, bias=True)
        )

        # use this as loss
        self.loss_func = nn.CrossEntropyLoss()


    def configure_optimizers(self):
        # create optimizer with our lr
        optimizer = torch.optim.AdamW(params=self.parameters(), lr=self.lr)
        return optimizer


    def forward(self, x):
        # pass x through net
        return self.net(x)


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

