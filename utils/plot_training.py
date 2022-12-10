

import argparse
import csv
import matplotlib.pyplot as plt


def main(args):
    filepath = args.path

    data = {}
    header = None

    with open(filepath, newline='') as csvfile:
        spamreader = csv.reader(csvfile, dialect='excel')
        for row in spamreader:

            if header is None:
                header = row
                for h in header:
                    data[h] = []
            
            else:
                for i in range(len(header)):
                    d = data[header[i]]
                    if row[i] == "":
                        d.append(None)
                    else:
                        d.append(float(row[i]))
    
    train_x = []
    for i in range(len(data['epoch'])):
        if data['train_acc'][i] is not None:
            train_x.append(data['epoch'][i])
    val_x = []
    for i in range(len(data['epoch'])):
        if not data['val_acc'][i] is None:
            val_x.append(data['epoch'][i])

    data['train_acc'] = list(filter(lambda x: x is not None, data['train_acc']))
    data['val_acc'] = list(filter(lambda x: x is not None, data['val_acc']))
    data['train_loss'] = list(filter(lambda x: x is not None, data['train_loss']))
    data['valid_loss'] = list(filter(lambda x: x is not None, data['valid_loss']))

    # plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(2, 1)

    ax[0].plot(train_x, data["train_loss"], 'r')
    ax[0].plot(val_x, data["valid_loss"], 'b')
    ax[0].legend(["train_loss", "val_loss"])
    ax[0].set_ylabel('Mean Binary Cross\nEntropy Loss')

    ax[1].plot(train_x, data["train_acc"], 'r')
    ax[1].plot(val_x, data["val_acc"], 'b')
    ax[1].legend(["train_acc", "val_acc"])
    ax[1].set_ylabel('Accuracy')

    plt.suptitle('Model Progress Through Training \n (With Gaussian Noise)')
    plt.xlabel('Epoch')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model to identify AI art.')

    parser.add_argument('-p', '--path', dest='path', type=str, default=None, 
                    help='path to csv')

    args = parser.parse_args()
    if args.path is None:
        raise ValueError("Please input path to checkpoint.")
    main(args)