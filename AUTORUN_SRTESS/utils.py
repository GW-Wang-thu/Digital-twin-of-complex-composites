import numpy as np
import matplotlib.pyplot as plt

def read_force(file_list, line, row):
    rec = []
    for i in range(len(file_list)):
        rec.append(np.load(file_list[i])[1, 2801])
    return rec


def main():
    rec = []
    file_name = r'E:\Data\DATASET\SealDigitTwin\FINAL\STRESS\input\force_label_1.txt'
    load_array = np.loadtxt(file_name, delimiter='\t')
    pressure_disp_force = np.array([load_array[:, 1], 100 - load_array[:, 0], load_array[:, 2]], dtype='float32').T
    for i in range(16):
        temp_label = np.load(r'E:\Data\DATASET\SealDigitTwin\FINAL\STRESS\input\BM_STRESS\BM_25_'+str(int(pressure_disp_force[i, 1]))+'_'+str(pressure_disp_force[i, -1])+"_STRESS.npy")
        rec.append(temp_label[1, 1839])
    plt.plot(rec)
    plt.show()


if __name__ == '__main__':
    main()
