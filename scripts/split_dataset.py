import random
import numpy as np

def train_val_split(img_file, ytrue_file, train_scale, val_scale):
    """Given a two files containing the list of images and ground truth path,
    Split them into train set and validation set.
    """
    with open(img_file) as imgs:
        img_names = imgs.read().splitlines()
    imgs.close()
    with open(ytrue_file) as ytrues:
        ytrue_names = ytrues.read().splitlines()
    ytrues.close()

    shuffled = list(zip(img_names, ytrue_names))
    random.shuffle(shuffled)
    img_names, ytrue_names = zip(*shuffled)

    train_end_idx = int(np.floor(len(img_names) * train_scale))
    val_end_idx =  int(np.floor(len(img_names) * (train_scale + val_scale)))

    assert len(img_names) == len(ytrue_names)
    # Generate the train set
    with open("img_train.txt", 'w') as img_train:
        img_train.write("\n". join(img_names[0:train_end_idx]))
    img_train.close()
    with open("ytrue_train.txt", 'w') as ytrue_train:
        ytrue_train.write("\n". join(ytrue_names[0:train_end_idx]))
    ytrue_train.close()
    # Generate the validation set
    with open("img_val.txt", 'w') as img_val:
        img_val.write("\n". join(img_names[train_end_idx:val_end_idx]))
    img_val.close()
    with open("ytrue_val.txt", 'w') as ytrue_val:
        ytrue_val.write("\n". join(ytrue_names[train_end_idx:val_end_idx]))
    ytrue_val.close()
    print("Training set and validation set splitted")

if __name__ == "__main__":
    train_val_split("images.txt", "labels.txt", 0.8, 0.2)