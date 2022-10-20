
import torch
import pytorch_lightning as pl

import argparse

from efficientnet_v2_custom import EFFICIENTNET_V2_CUSTOM
from binarydataset import BinaryDataset

def main(args):

    dataset = BinaryDataset("./non_AI_art/", "./_digital/", max_size=10)
    dataset.to(args.device)
    train_loader = torch.utils.data.DataLoader(dataset, shuffle=True, )

    model = EFFICIENTNET_V2_CUSTOM()
    model.to(args.device)
    optimizer = model.configure_optimizers()

    model.train()

    logger = pl.loggers.CSVLogger(save_dir='lightning_logs', flush_logs_every_n_steps=10000)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath="checkpoints", save_top_k=5, monitor="train_loss")

    trainer = pl.Trainer(logger=logger, log_every_n_steps=1, callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model to identify AI art.')

    parser.add_argument('--gpu', dest='device', action='store_const', const=torch.device('cuda'), default=torch.device('cpu'), 
                    help='Whether to use cuda gpu acceleration')

    args = parser.parse_args()
    main(args)