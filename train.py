
import torch
import pytorch_lightning as pl

from efficientnet_v2_custom import EFFICIENTNET_V2_CUSTOM
from binarydataset import BinaryDataset

def main():

    dataset = BinaryDataset("./training_images/nebulas/", "./training_images/nebulas/")
    train_loader = torch.utils.data.DataLoader(dataset)

    model = EFFICIENTNET_V2.EFFICIENTNET_V2_CUSTOM()
    optimizer = model.configure_optimizers()

    logger = pl.loggers.CSVLogger(flush_logs_every_n_steps=10000)

    trainer = pl.Trainer(logger=logger, log_every_n_steps=1)
    trainer.fit(model, train_loader)

if __name__ == '__main__':
    main()