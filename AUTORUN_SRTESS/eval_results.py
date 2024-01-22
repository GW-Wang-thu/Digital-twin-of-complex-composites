import numpy as np
import matplotlib.pyplot as plt


def plot_stress(stress_csv, label_npy, coord_npy, sample_id, option=0):
    recon_stress_file = np.loadtxt(stress_csv, delimiter=",")
    stress_predicted = recon_stress_file[sample_id, :]
    names = label_npy.split("\\")[-1].split('.')[0].split('_')
    stress_label = np.load(label_npy)[option, :]
    coord_vect = np.load(coord_npy)
    stress_error = stress_predicted - stress_label
    mse = np.mean(stress_error**2)
    print("MSE: ", mse)

    x_coord = [coord_vect[3 * i] for i in range(coord_vect.shape[0] // 3)]
    y_coord = [coord_vect[3 * i + 1] for i in range(coord_vect.shape[0] // 3)]
    z_coord = [coord_vect[3 * i + 2] for i in range(coord_vect.shape[0] // 3)]

    if option==0:
        namestring = "s11"
    elif option == 1:
        namestring = 's22'
    elif option == 2:
        namestring = 's33'
    elif option == 3:
        namestring = 's12'
    elif option == 4:
        namestring = 's13'
    elif option == 5:
        namestring = 's23'

    s_l = stress_label.tolist()
    s = stress_predicted.tolist()
    s_e = stress_error.tolist()
    vmin = min(s_l + s)
    vmax = max(s_l + s)
    fig = plt.figure(dpi=128, figsize=(22, 8))
    ax = fig.add_subplot(131, projection='3d')
    p1 = ax.scatter(x_coord, y_coord, z_coord, s=2, cmap="jet", c=s, label="decoded " + namestring)
    p1.set_clim(vmin, vmax)
    fig.colorbar(p1, fraction=0.045, pad=0.05)
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)
    plt.title("Predicted Stress " + namestring + " of T: %d deg, P: %d kPa, D: %.2f mm" % (
        int(names[1]), int(names[2]), float(names[3])))
    plt.legend()

    ax2 = fig.add_subplot(132, projection='3d')
    p2 = ax2.scatter(x_coord, y_coord, z_coord, s=2, cmap="jet", c=s_l, label="labeled " + namestring)
    p2.set_clim(vmin, vmax)
    fig.colorbar(p2, fraction=0.045, pad=0.05)
    ax2.set_xlabel('X', fontsize=10)
    ax2.set_ylabel('Y', fontsize=10)
    ax2.set_zlabel('Z', fontsize=10)
    plt.title("Labeled Stress " + namestring + " of T: %d deg, P: %d kPa, D: %.2f mm" % (
        int(names[1]), int(names[2]), float(names[3])))
    plt.legend()

    ax3 = fig.add_subplot(133, projection='3d')
    p3 = ax3.scatter(x_coord, y_coord, z_coord, s=2, cmap="jet", c=s_e, label="ERROR " + namestring)
    p3.set_clim(min(s_e), max(s_e))
    fig.colorbar(p3, fraction=0.045, pad=0.05)
    ax3.set_xlabel('X', fontsize=10)
    ax3.set_ylabel('Y', fontsize=10)
    ax3.set_zlabel('Z', fontsize=10)
    plt.title("Stress Error " + namestring + " MSE = %.5f" % (float(mse)))
    plt.legend()

    plt.show()


if __name__ == '__main__':
    disp=20.6
    for i in range(18):
        plot_stress(stress_csv=r"E:\Data\DATASET\SealDigitTwin\FINAL\Fig2_STRESS\results\\"+str(i)+"_Train_20.6_BM_Bayesian_s22.csv",
                    label_npy=r"E:\Data\DATASET\SealDigitTwin\Results\0_input_npy\STRESS\BM_25_0_"+str(disp)+"_STRESS.npy",
                    coord_npy=r"E:\Data\DATASET\SealDigitTwin\Results\0_input_npy\COORD\BM_25_0_"+str(disp)+"_COORD.npy",
                    sample_id=15,
                    option=1)

