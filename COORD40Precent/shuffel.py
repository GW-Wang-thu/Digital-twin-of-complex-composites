import numpy as np
import os
import shutil


def shuffel():
    train_dir = r'E:\Data\DATASET\SealDigitTwin\COORD\TRAIN\\'
    to_dir = r'E:\Data\DATASET\SealDigitTwin\40PTRAIN\TRAIN\\'
    allfiles = os.listdir(train_dir)
    all_BM_coord = [os.path.join(train_dir, file) for file in allfiles if file.endswith("Coord.npy") and file.startswith("BM")]
    all_to_BM_coord = [os.path.join(to_dir, file) for file in allfiles if file.endswith("Coord.npy") and file.startswith("BM")]

    for i in range(len(all_BM_coord)):
        if np.random.rand() > 0.5:
            shutil.copy(all_BM_coord[i], all_to_BM_coord[i])


if __name__ == '__main__':
    shuffel()