import shutil
import os
import numpy as np


def random_delete(percent, init_dir, to_dir):
    all_files = os.listdir(init_dir)
    all_npys = [os.path.join(init_dir, file) for file in all_files if file.endswith(".npy")]
    for i in range(len(all_npys)):
        if np.random.rand() > percent:
            shutil.move(all_npys[i], os.path.join(to_dir, all_files[i]))


if __name__ == '__main__':
    random_delete(0.4, init_dir=r"E:\Data\DATASET\SealDigitTwin\GPDomainLearn\TRAIN//",
                  to_dir=r"E:\Data\DATASET\SealDigitTwin\GPDomainLearn\TRAIN_G")