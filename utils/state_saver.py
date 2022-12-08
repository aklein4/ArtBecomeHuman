
import torch

CHECKPOINT_FOLDER = "./checkpoints/"
TO_SAVE = [
    'epoch=288-step=58089',
    'epoch=296-step=59697',
    'epoch=302-step=60903',
    'epoch=325-step=65526',
    'epoch=327-step=65928',
]

SAVE_FOLDER = "./model_states/"

def main():
    for filename in TO_SAVE:
        checkpoint = torch.load(CHECKPOINT_FOLDER+filename+'.ckpt')
        torch.save(checkpoint['state_dict'], SAVE_FOLDER+filename+'.pt')


if __name__ == '__main__':
    main()