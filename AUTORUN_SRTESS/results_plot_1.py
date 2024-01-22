import os.path

import numpy as np
import matplotlib.pyplot as plt


def normfun(x, mu, sigma):
    pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return pdf


def get_data(filename, idx, format):
    if format=='txt':
        data = np.loadtxt(filename, delimiter=',') # 载入数据文件
    elif format=='npy':
        data = np.load(filename)
    length = data[:, idx]
    mean = length.mean() # 获得数据集的平均值
    std = length.std()   # 获得数据集的标准差
    return length, mean, std, np.mean(data, axis=0)



def plot_distribution(filename_list, idx, label):
    results_rec = []
    mini = 100
    maxi = -100
    for i in range(len(filename_list)):
        temp_result = get_data(filename_list[i], idx, format='npy')
        temp_result = temp_result[:-1]
        results_rec.append(temp_result)
        mini = min(mini, np.min(temp_result[0]))
        maxi = max(maxi, np.max(temp_result[0]))

    x_lim = np.linspace(mini, maxi+0.1, num=1000)
    for i in range(len(filename_list)):
        plt.plot(x_lim, normfun(x_lim, results_rec[i][1], results_rec[i][2]) * results_rec[i][2]**0.5, label="T"+str(i))
    plt.plot([label for j in range(1000)], normfun(x_lim, results_rec[i][1], results_rec[i][2]) * results_rec[i][2] ** 0.5, ":", label="Label Value")
    plt.legend()
    plt.show()
    plt.plot([mean[1] for mean in results_rec], label="mean")
    plt.plot([mean[2] for mean in results_rec], label="std")
    plt.legend()
    # plt.show()


def plot_bnd(filename_list, idx, label_rec=None, title=None):
    mean_rec = []
    std_rec = []

    for i in range(len(filename_list)):
        _, mean, std, _ = get_data(filename_list[i], idx, format='npy')
        mean_rec.append(mean)
        std_rec.append(std)

    mean_array = np.array([mean_rec]).T
    std_array = np.array([std_rec]).T
    truth_array = np.array([label_rec]).T

    plt.plot(mean_rec, label="mean")
    if label_rec is not None:
        plt.plot(label_rec, label="label")
    plt.fill_between(range(len(mean_rec)),
                     [mean_rec[i]-std_rec[i]*2 for i in range(len(mean_rec))],
                     [mean_rec[i]+std_rec[i]*2 for i in range(len(mean_rec))],
                     color="b",
                     alpha=0.1
                     )
    plt.legend()
    plt.title(title)
    plt.show()


def plot_error(filename_list, label, title=None):
    error_rec = []
    for i in range(len(filename_list)):
        try:
            _, _, _, mean = get_data(filename_list[i], 0, format='npy')
        except:
            pass
        error_rec.append(np.average((mean-label)**2))
    error_array = np.array([error_rec]).T
    np.savetxt("error.txt", error_array)
    err = error_array.T
    plt.plot(error_rec)
    plt.xlabel("update step")
    plt.ylabel("MSE")
    plt.title(title)
    plt.show()


def eval_bnd():
    # label = np.loadtxt(r"E:\Data\DATASET\SealDigitTwin\FINAL\Fig2_STRESS\results\\label_stress.txt", delimiter='\t')
    # fname_list = [
    #     r"E:\Data\DATASET\SealDigitTwin\FINAL\Fig2_STRESS\results_1\\0_20.6_BM_Initial_s22.csv",
    #     r"E:\Data\DATASET\SealDigitTwin\FINAL\Fig2_STRESS\results_1\\0_20.6_BM_Bayesian_s22.csv",
    # ]
    # for idx in range(1, 15000, 500):

    # idx=1500
    # fname_list = []
    # for step in range(18):
    #     fname_list.append(r"E:\Data\DATASET\SealDigitTwin\FINAL\Fig2_STRESS\results\\"+str(step)+"_Train_20.6_BM_Bayesian_s22.csv")
    # plot_distribution(filename_list=fname_list, idx=idx, label=label[idx, 1])
    # label_dir = r'E:\Data\DATASET\SealDigitTwin\FINAL\STRESS\input\BM_STRESS\\'
    load_array = np.loadtxt(r"E:\Data\DATASET\SealDigitTwin\FINAL\STRESS\input\\load_stress.txt", delimiter='\t')
    # pressure_disp_force = np.array([load_array[:, 1], 100 - load_array[:, 0], load_array[:, 2]], dtype='float32').T
    rec = []
    time = 46
    idx = 1000
    component = 5
    label_rec = []
    string = 's23'
    for i in range(58):
        dir_0 = r'E:\Data\DATASET\SealDigitTwin\FINAL\STRESS\results_10\\' + str(i) + '\\' + str(
            time) + "_20221021_BM_Bayesian_"+string+".npy"
        dir_1 = r'E:\Data\DATASET\SealDigitTwin\FINAL\STRESS\results_10\\' + str(i) + '\\' + str(
            time) + "_20221021_BM_Bayesian_"+string+".npy"
        if os.path.exists(dir_0):
            rec.append(dir_0)
        else:
            rec.append(dir_1)
        temp_label = np.load(r'E:\Data\DATASET\SealDigitTwin\FINAL\STRESS\input\BM_STRESS\BM_25_'+str(int(load_array[i, 0]))+'_'+str(load_array[i, -1])+"_STRESS.npy")
        label_rec.append(temp_label[component, idx])
    # plot_distribution(filename_list=rec, idx=1000, label=label[1000, 0])
    plot_bnd(rec, idx, label_rec, title="Time = "+str(time)+" "+string)


def eval_error():
    load_array = np.loadtxt(r'E:\Data\DATASET\SealDigitTwin\FINAL\STRESS\input\\load_stress.txt', delimiter='\t')
    # pressure_disp_force = np.array([load_array[:, 1], 100 - load_array[:, 0], load_array[:, 2]], dtype='float32').T
    rec = []
    sample_id = 40
    component = 0
    string = 's11'
    for i in range(58):
        dir_0 = r'E:\Data\DATASET\SealDigitTwin\FINAL\STRESS\results_10\\' + str(sample_id) + '\\' + str(
            i) + "_20221021_BM_Bayesian_"+string+".npy"
        dir_1 = r'E:\Data\DATASET\SealDigitTwin\FINAL\STRESS\results_10\\' + str(sample_id) + '\\' + str(
            i) + "_20221021_BM_Bayesian_"+string+".npy"
        if os.path.exists(dir_0):
            rec.append(dir_0)
        else:
            rec.append(dir_1)
    label = np.load(r'E:\Data\DATASET\SealDigitTwin\FINAL\STRESS\input\BM_STRESS\BM_25_'+str(int(load_array[sample_id, 0]))+'_'+str(load_array[sample_id, -1])+"_STRESS.npy")
    # plot_distribution(filename_list=rec, idx=1000, label=label[1000, 0])
    print("Pressure: %.2f, displacement: %.2f"%(load_array[sample_id, 0], load_array[sample_id, -1]))
    plot_error(rec, label=label[component], title="Sample ID=%d"%(sample_id)+" "+string+" MSE")


def eval_distribution():
    load_array = np.loadtxt(r'E:\Data\DATASET\SealDigitTwin\FINAL\STRESS\input\\load_stress.txt', delimiter='\t')
    sample_id = 40
    component = 1
    string = 's22'
    idx = 1000
    label = np.load(r'E:\Data\DATASET\SealDigitTwin\FINAL\STRESS\input\BM_STRESS\BM_25_'+str(int(load_array[sample_id, 0]))+'_'+str(load_array[sample_id, -1])+"_STRESS.npy")
    label_value = label[component, idx]

    coord_idx = [0, 14, 18, 44]
    name_rec = []
    for i in range(len(coord_idx)):
        dir_1 = r'E:\Data\DATASET\SealDigitTwin\FINAL\STRESS\results_10\\' + str(sample_id) + '\\' + str(coord_idx[i]) + "_20221021_BM_Bayesian_"+string+".npy"
        name_rec.append(dir_1)

    plot_distribution(name_rec, idx, label_value)


if __name__ == '__main__':
    # eval_bnd()
    # eval_error()
    eval_distribution()