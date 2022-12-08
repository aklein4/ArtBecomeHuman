
import torch
import pytorch_lightning as pl

import argparse
import os

from efficientnet_v2_custom import EFFICIENTNET_V2_CUSTOM
from binarydataset import BinaryDataset
from pytorch_lightning.loggers import CSVLogger

REAL_PATH = "/home/adam3207/data/artbench"
AI_PATH = "/home/adam3207/data/__AI__artbench"


def main(args):

    # load training data
    print("\nloading training data...")
    train_data = BinaryDataset(os.path.join(REAL_PATH, "train/"), os.path.join(AI_PATH, "train/"), skip_len=args.skip)
    train_loader = torch.utils.data.DataLoader(
        train_data, shuffle=True, batch_size=args.batchsize, num_workers=12
    )
    print("Training Data Sizes: Real -", torch.sum(train_data.labels[:, 0]).item(), "AI -", torch.sum(train_data.labels[:, 1]).item())

    # load training data
    print("\nloading validation data...")
    val_data = BinaryDataset(os.path.join(REAL_PATH, "test/"), os.path.join(AI_PATH, "test/"), skip_len=args.skip)
    val_loader = torch.utils.data.DataLoader(
        val_data, shuffle=False, batch_size=args.batchsize, num_workers=12
    )
    print("Training Data Sizes: Real -", torch.sum(val_data.labels[:, 0]).item(), "AI -", torch.sum(val_data.labels[:, 1]).item())
    
    print("")

    # create model
    model = EFFICIENTNET_V2_CUSTOM(lr=args.lr, n_classes=2, pretrained=True)
    model.to(args.device)
    model.train()

    # initialize logger
    logger = CSVLogger(
        save_dir='.',
        flush_logs_every_n_steps=100000
    )

    # callback for best model
    best_callback = pl.callbacks.ModelCheckpoint(
        dirpath="best_checkpoints", save_top_k=5,
        monitor="valid_loss", mode="min",
    )

    # run training
    trainer = pl.Trainer(
        accelerator=('gpu' if args.device == torch.device('cuda') else 'cpu'),
        logger=logger,
        check_val_every_n_epoch=1,
        callbacks=[best_callback], max_epochs=-1
        )

    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model to identify AI art.')

    parser.add_argument('--gpu', dest='device', action='store_const', const=torch.device('cuda'), default=torch.device('cpu'), 
                    help='Whether to use cuda gpu acceleration')

    parser.add_argument('-bs', '--batchsize', dest='batchsize', type=int, default=128, 
                    help='Whether to use cuda gpu acceleration')
    
    parser.add_argument('-lr', '--learningrate', dest='lr', type=float, default=1e-5, 
                    help='Whether to use cuda gpu acceleration')
    
    parser.add_argument('-s', '--skip', dest='skip', type=int, default=1, 
                    help='Divide the training and val sets by this size.')
    

    args = parser.parse_args()
    main(args)
