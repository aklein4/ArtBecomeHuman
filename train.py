
import torch
import pytorch_lightning as pl

import argparse

from efficientnet_v2_custom import EFFICIENTNET_V2_CUSTOM
from binarydataset import BinaryDataset

def main(args):

    # load training data
    train_data = BinaryDataset("./non_AI_art/", "./_digital/", max_size=10)
    train_data.to(args.device)
    train_loader = torch.utils.data.DataLoader(
        train_data, shuffle=True, batch_size=args.batchsize
    )

    # load validation data
    val_data = BinaryDataset("./non_AI_art/", "./_digital/", max_size=10)
    val_data.to(args.device)
    val_loader = torch.utils.data.DataLoader(
        val_data, shuffle=False, batch_size=args.batchsize
    )

    # create model
    model = EFFICIENTNET_V2_CUSTOM(args.lr)
    model.to(args.device)
    optimizer = model.configure_optimizers()
    model.train()

    # initialize extra stuff
    logger = pl.loggers.CSVLogger(
        save_dir='lightning_logs',
        flush_logs_every_n_steps=10000
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints", save_top_k=5,
        monitor="valid_loss"
    )

    # run training
    trainer = pl.Trainer(
        logger=logger, log_every_n_steps=1,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback]
        )
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model to identify AI art.')

    parser.add_argument('--gpu', dest='device', action='store_const', const=torch.device('cuda'), default=torch.device('cpu'), 
                    help='Whether to use cuda gpu acceleration')
    parser.add_argument('-bs', '--batchsize', dest='batchsize', type=int, default=1, 
                    help='Whether to use cuda gpu acceleration')
    parser.add_argument('-lr', '--learningrate', dest='lr', type=float, default=1e-5, 
                    help='Whether to use cuda gpu acceleration')

    args = parser.parse_args()
    main(args)
