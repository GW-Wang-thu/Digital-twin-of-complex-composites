import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def normfun(x, mu, sigma):
    pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return pdf

def get_data(filename, idx, format='txt'):
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


def plot_bnd(filename_list, idx, label_rec=None, title=None, xlabel="Time Step", ylabel="COORD(mm)", save=True):
    mean_rec = []
    std_rec = []

    plt.figure(figsize=(15, 9.5))
    for i in range(len(filename_list)):
        temp_mean_rec = []
        temp_std_rec = []
        for j in range(len(filename_list[i])):
            _, mean, std, _ = get_data(filename_list[i][j], idx, format='npy')
            temp_mean_rec.append(mean)
            temp_std_rec.append(std)
        mean_rec.append(temp_mean_rec)
        std_rec.append(temp_std_rec)

    for j in range(len(filename_list)):
        plt.subplot(len(filename_list), 1, j+1)
        plt.plot(mean_rec[j], label="mean")
        if label_rec is not None:
            plt.plot(label_rec[j], label="label")
        if title is not None:
            plt.title(title[j])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    # mean_array = np.array([mean_rec]).T
    # std_array = np.array([std_rec]).T
    # truth_array = np.array([label_rec]).T
        plt.fill_between(range(len(mean_rec[j])),
                         [mean_rec[j][i]-std_rec[j][i]*2 for i in range(len(mean_rec[j]))],
                         [mean_rec[j][i]+std_rec[j][i]*2 for i in range(len(mean_rec[j]))],
                         color="b",
                         alpha=0.1
                         )
        plt.legend()
    if save:
        tmp_name = filename_list[0][0].split("\\")[-1].split("_")[0] + ".png"
        plt.savefig("./pngs/"+tmp_name)
    plt.show()




def plot_error(filename_list, label, title=None):
    error_rec = []
    for i in range(len(filename_list)):
        try:
            _, _, _, mean = get_data(filename_list[i], 0, format='npy')
        except:
            print("ERROR")
            pass
        error_rec.append(np.average((mean-label)**2))
    error_array = np.array([error_rec]).T
    np.savetxt("error.txt", error_array)
    plt.plot(error_rec)
    plt.xlabel("update step")
    plt.ylabel("MSE")
    plt.title(title)
    plt.show()


def eval_bnd(time):
    rec = []
    load_array = np.loadtxt(r"E:\Data\DATASET\SealDigitTwin\FINAL\STRESS\input\\load_stress.txt", delimiter='\t')
    # pressure_disp_force = np.array([load_array[:, 1], 100 - load_array[:, 0], load_array[:, 2]], dtype='float32').T
    u_rec = []
    v_rec = []
    idx = 1000
    string = 'X'
    label_rec_u = []
    label_rec_v = []
    for i in range(58):
        u_rec.append(r'E:\Data\DATASET\SealDigitTwin\FINAL\COORD\results\L1\\'+str(i)+'\\' + str(time) + "_20221021_BM_Bayesian_Coord_Predicted_X.npy")
        v_rec.append(r'E:\Data\DATASET\SealDigitTwin\FINAL\COORD\results\L1\\'+str(i)+'\\' + str(time) + "_20221021_BM_Bayesian_Coord_Predicted_Y.npy")
        temp_label = np.load(r'E:\Data\DATASET\SealDigitTwin\FINAL\COORD\add_coords\BM_25_' + str(int(load_array[i, 0])) + '_' + str(load_array[i, -1]) + "_Coord.npy")
        temp_label = temp_label.reshape((temp_label.shape[0]//3, 3))
        label_rec_u.append(temp_label[idx, 0])
        label_rec_v.append(temp_label[idx, 1])
    # plot_distribution(filename_list=rec, idx=1000, label=label[1000, 0])
    plot_bnd([u_rec, v_rec], idx=idx,
             label_rec=[label_rec_u, label_rec_v],
             title=["Predicted U using model updated at time step "+str(time),
                    "Predicted V using model updated at time step "+str(time)],
             xlabel="Time Step",
             ylabel="COORD(mm)")


def eval_mse():
    load_array = np.loadtxt(r"E:\Data\DATASET\SealDigitTwin\FINAL\STRESS\input\\load_stress.txt", delimiter='\t')
    # pressure_disp_force = np.array([load_array[:, 1], 100 - load_array[:, 0], load_array[:, 2]], dtype='float32').T
    rec = []
    time = 46
    component = 1
    string = 'Y'
    error_rec = []
    for i in range(58):
        rec.append(r'E:\Data\DATASET\SealDigitTwin\FINAL\COORD\results\L1\\'+str(i)+'\\' + str(time) + "_20221021_BM_Bayesian_Coord_Predicted_"+string+".npy")
        temp_label = np.load(r'E:\Data\DATASET\SealDigitTwin\FINAL\COORD\add_coords\BM_25_' + str(int(load_array[i, 0])) + '_' + str(load_array[i, -1]) + "_Coord.npy")
        temp_label = temp_label.reshape((temp_label.shape[0]//3, 3))[:, component]

        _, _, _, mean = get_data(rec[i], idx=0, format='npy')
        error_rec.append(((mean - temp_label)**2).mean())
    print("MSE of L1: ", np.mean(error_rec))
    plt.plot(error_rec, label="mean")
    plt.title("MSE of "+string)
    plt.show()

def eval_error():
    load_array = np.loadtxt(r'E:\Data\DATASET\SealDigitTwin\FINAL\STRESS\input\\load_stress.txt', delimiter='\t')
    # pressure_disp_force = np.array([load_array[:, 1], 100 - load_array[:, 0], load_array[:, 2]], dtype='float32').T
    rec = []
    sample_id = 40
    component = 1
    string = 'Y'
    for i in range(58):
        dir_1 = r'E:\Data\DATASET\SealDigitTwin\FINAL\COORD\results\L1\\' + str(sample_id) + '\\' + str(i) + "_20221021_BM_Bayesian_Coord_Predicted_"+string+".npy"
        rec.append(dir_1)
    label = np.load(r'E:\Data\DATASET\SealDigitTwin\FINAL\COORD\add_coords\BM_25_' + str(int(load_array[sample_id, 0])) + '_' + str(load_array[sample_id, -1]) + "_Coord.npy")
    label = label.reshape((label.shape[0]//3, 3))
    # plot_distribution(filename_list=rec, idx=1000, label=label[1000, 0])
    print("Pressure: %.2f, displacement: %.2f"%(load_array[sample_id, 0], load_array[sample_id, -1]))
    plot_error(rec, label=label[:, component], title="Sample ID=%d"%(sample_id)+" "+string+" MSE")


def eval_distribution():
    load_array = np.loadtxt(r'E:\Data\DATASET\SealDigitTwin\FINAL\STRESS\input\\load_stress.txt', delimiter='\t')
    sample_id = 40
    component = 1
    string = 'Y'
    idx = 1000
    label = np.load(r'E:\Data\DATASET\SealDigitTwin\FINAL\COORD\add_coords\BM_25_' + str(int(load_array[sample_id, 0])) + '_' + str(load_array[sample_id, -1]) + "_Coord.npy")
    label = label.reshape((label.shape[0] // 3, 3))
    label_value = label[idx, component]

    coord_idx = [0, 14, 18, 44]
    name_rec = []
    for i in range(len(coord_idx)):
        dir_1 = r'E:\Data\DATASET\SealDigitTwin\FINAL\COORD\results\L1\\' + str(sample_id) + '\\' + str(coord_idx[i]) + "_20221021_BM_Bayesian_Coord_Predicted_" + string + ".npy"
        name_rec.append(dir_1)

    plot_distribution(name_rec, idx, label_value)


if __name__ == '__main__':
    for time in range(58):
        eval_bnd(time)
    # eval_mse()
    # eval_error()
    # eval_distribution()