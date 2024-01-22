import numpy as np
import matplotlib.pyplot as plt


def normfun(x, mu, sigma):
    pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return pdf

def get_data(filename, idx):
    data = np.loadtxt(filename, delimiter=',') # 载入数据文件
    length = data[:, idx]
    mean = length.mean() # 获得数据集的平均值
    std = length.std()   # 获得数据集的标准差
    return length, mean, std

def plot_distribution(filename_list, idx, label):
    results_rec = []
    mini = 100
    maxi = -100
    for i in range(len(filename_list)):
        temp_result = get_data(filename_list[i], idx)
        results_rec.append(temp_result)
        mini = min(mini, np.min(temp_result[0]))
        maxi = max(maxi, np.max(temp_result[0]))

    x_lim = np.linspace(mini, maxi, num=1000)
    for i in range(len(filename_list)):
        plt.plot(x_lim, normfun(x_lim, results_rec[i][1], results_rec[i][2]) * results_rec[i][2]**0.5, label="T"+str(i))
    plt.plot([label for j in range(1000)], normfun(x_lim, results_rec[i][1], results_rec[i][2]) * results_rec[i][2] ** 0.5, ":", label="Label Value")
    plt.legend()
    plt.show()
    plt.plot([mean[1] for mean in results_rec], label="mean")
    plt.plot([mean[2] for mean in results_rec], label="std")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    label = np.loadtxt(r"E:\Data\DATASET\SealDigitTwin\FINAL\Fig2\results\case\\label_coord.txt", delimiter='\t')
    plot_distribution(filename_list=[
        r"E:\Data\DATASET\SealDigitTwin\FINAL\Fig2\\results\case\T0\\0_T1_BM_pretrain_Coord_Predicted_X.csv",
        r"E:\Data\DATASET\SealDigitTwin\FINAL\Fig2\\results\case\T1\\0_T1_BM_aftertrain_Coord_Predicted_X.csv",
        r"E:\Data\DATASET\SealDigitTwin\FINAL\Fig2\\results\case\T2\\0_T2_BM_aftertrain_Coord_Predicted_X.csv",
        r"E:\Data\DATASET\SealDigitTwin\FINAL\Fig2\\results\case\T3\\0_T3_BM_aftertrain_Coord_Predicted_X.csv",
        r"E:\Data\DATASET\SealDigitTwin\FINAL\Fig2\\results\case\T4\\0_T4_BM_aftertrain_Coord_Predicted_X.csv",
    ], idx=1000, label=label[1000, 0])