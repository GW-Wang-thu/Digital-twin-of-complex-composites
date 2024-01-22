import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import numpy as np
from STRESS.network import Predictor, VAE
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt


class STRESS_dataloader(Dataset):
    def __init__(self, file_dir, type="BM", ratio_vect=[4.0, 1.0, 4.0, 10.0, 20.0, 8.0]):
        self.all_files = os.listdir(file_dir)
        self.all_inputs = [os.path.join(file_dir, file) for file in self.all_files if (file.endswith("STRESS.npy") and file.startswith(type))]
        self.type = type
        self.ratio_vect = torch.from_numpy(np.array(ratio_vect, dtype="float32")).cuda()

    def __len__(self):
        return len(self.all_inputs)

    def __getitem__(self, item):
        random_array = np.random.normal(1, 0.05, size=(6,)).astype("float32")
        filename = self.all_inputs[item]
        vect = np.load(filename)
        vect_torch = torch.from_numpy(vect).cuda()
        vect_torch_m = torch.transpose(torch.transpose(vect_torch, 0, 1) * self.ratio_vect * torch.from_numpy(random_array).cuda(), 0, 1)
        return vect_torch_m, filename


class PreDataloader(Dataset):
    def __init__(self, file_dir, type="BM", ratio_vect=[4.0, 1.0, 4.0, 10.0, 20.0, 8.0]):
        self.all_files = os.listdir(file_dir)
        self.all_names = [file for file in self.all_files if (file.endswith("STRESS.npy") and file.startswith(type))]
        self.all_inputs = [os.path.join(file_dir, file) for file in self.all_files if (file.endswith("STRESS.npy") and file.startswith(type))]
        self.ratio_vect = torch.from_numpy(np.array(ratio_vect, dtype="float32")).cuda()

    def __len__(self):
        return len(self.all_inputs)

    def __getitem__(self, item):
        filename = self.all_inputs[item]
        name = self.all_names[item]
        # print(name)
        name_list = name.split("_")
        temperature = int(name_list[1])
        pressure = int(name_list[2])
        displacement = float(name_list[3])
        vect = np.load(filename)
        vect_torch = torch.from_numpy(vect).cuda()
        vect_torch_m = torch.transpose(torch.transpose(vect_torch, 0, 1) * self.ratio_vect, 0, 1)
        return torch.from_numpy(np.array([temperature, pressure, displacement], dtype="float32")).cuda(), vect_torch_m, filename


def train_autocoder(type="BM", save=False, display=False, step=10):
    trainset_dir = r"E:\Data\DATASET\SealDigitTwin\TransLearn\NPY\TRAIN\\"
    validset_dir = r"E:\Data\DATASET\SealDigitTwin\TransLearn\NPY\VALID\\"
    ratio_train = True
    if ratio_train:
        ratio_vect = [4.5, 1.0, 4.5, 12.0, 25.0, 8.0]
    else:
        ratio_vect = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    if save:
        savedir = "I:\\DigitRubber_Dataset\\NPY\\VALID_RESULTS\\COORD\\CODER\\"
    train_dl = DataLoader(STRESS_dataloader(trainset_dir, type=type, ratio_vect=ratio_vect),
                          batch_size=1,
                          shuffle=True)
    valid_dl = DataLoader(STRESS_dataloader(validset_dir, type=type, ratio_vect=ratio_vect),
                          batch_size=1,
                          shuffle=True)

    lr = 1e-4
    if type == "BM":
        vect_length = 15184
    elif type == "NB":
        vect_length = 13120
    else:
        vect_length = 26016

    autocoder = VAE(vect_length=vect_length, code_length=72, num_layer=2, in_channel=6).cuda()

    if os.path.exists("./params1-noshuffle/" + type + "_coder_last.pth"):
        checkpoint = torch.load("./params1-noshuffle/" + type + "_coder_last.pth")
        autocoder.load_state_dict(checkpoint)
        results_rec = np.loadtxt("./params1-noshuffle/" + type + "_training_rec.txt")
        loss_rec_all_valid = results_rec[:, 1].tolist()
        loss_rec_all_train = results_rec[:, 2].tolist()
        results_rec = results_rec.tolist()
    else:
        loss_rec_all_valid = [10000]
        loss_rec_all_train = [10000]
        results_rec = []

    optimizer = torch.optim.Adam(autocoder.parameters(), lr=lr)
    epoch_n = 1000

    loss_rec = []
    for i in range(epoch_n-len(results_rec)*step):
        if i == 0:
            epoch = len(results_rec)*step
        epoch += 1
        autocoder.train()

        if epoch % 200 == 199:
            lr = lr / 2
            optimizer = torch.optim.Adam(autocoder.parameters(), lr=lr)

        for i, (vects, _) in enumerate(train_dl):
            autocoder.zero_grad()
            recon, mean, log_std = autocoder(vects.clone())
            loss, recon_loss = autocoder.loss_function(recon, vects, mean, log_std)
            loss.backward()
            optimizer.step()
            loss_rec.append(recon_loss.item())

        if epoch % step == 0 or epoch == epoch_n - 1:
            train_loss_mean = np.mean(np.array(loss_rec))
            loss_rec_all_train.append(train_loss_mean)
            loss_rec = []
            for i, (vects, file_name) in enumerate(valid_dl):
                recon, mean, log_std = autocoder(vects.clone())
                loss, recon_loss = autocoder.loss_function(recon, vects, mean, log_std)
                loss_rec.append(recon_loss.item())

                '''下面的代码是用于可视化的，训练的时候可以注释掉'''
                if display:
                    vect_outputs = recon[0, :].detach().cpu().numpy()
                    vect_init = vects[0, :].detach().cpu().numpy()
                    input_names = file_name[0].split("\\")[-1].split("_")

                    s11 = vect_outputs[0, :].tolist()
                    s22 = vect_outputs[1, :].tolist()
                    s33 = vect_outputs[2, :].tolist()
                    s12 = vect_outputs[3, :].tolist()
                    s13 = vect_outputs[4, :].tolist()
                    s23 = vect_outputs[5, :].tolist()

                    s11_l = vect_init[0, :].tolist()
                    s22_l = vect_init[1, :].tolist()
                    s33_l = vect_init[2, :].tolist()
                    s12_l = vect_init[3, :].tolist()
                    s13_l = vect_init[4, :].tolist()
                    s23_l = vect_init[5, :].tolist()

                    temp_coord_filename = "E:\Data\DATASET\SealDigitTwin\\\COORD\\TRAIN\\" + file_name[0].split("\\")[-1][
                                                                                           :-10] + "COORD.npy"
                    try:
                        coord_vect = np.load(temp_coord_filename)
                    except:
                        temp_coord_filename = "E:\Data\DATASET\SealDigitTwin\\\COORD\\VALID\\" + file_name[0].split("\\")[
                                                                                                   -1][
                                                                                               :-10] + "COORD.npy"
                        coord_vect = np.load(temp_coord_filename)

                    x_coord = [coord_vect[3 * i] for i in range(coord_vect.shape[0] // 3)]
                    y_coord = [coord_vect[3 * i + 1] for i in range(coord_vect.shape[0] // 3)]
                    z_coord = [coord_vect[3 * i + 2] for i in range(coord_vect.shape[0] // 3)]

                    #
                    ratio = ratio_vect[0]
                    s = [s / ratio for s in s11]
                    s_l = [s / ratio for s in s11_l]
                    namestring = "s11"

                    vmin = min(s_l + s)
                    vmax = max(s_l + s)
                    fig = plt.figure(dpi=128, figsize=(16, 8))
                    ax = fig.add_subplot(121, projection='3d')
                    p1 = ax.scatter(x_coord, y_coord, z_coord, s=2, cmap="jet", c=s, label="decoded " + namestring)
                    p1.set_clim(vmin, vmax)
                    fig.colorbar(p1, fraction=0.045, pad=0.05)
                    ax.set_xlabel('X', fontsize=10)
                    ax.set_ylabel('Y', fontsize=10)
                    ax.set_zlabel('Z', fontsize=10)
                    plt.title("Predicted Stress " + namestring + " of T: %d deg, P: %d kPa, D: %.2f mm" % (
                    int(input_names[1]), int(input_names[2]), float(input_names[3])))
                    plt.legend()

                    ax2 = fig.add_subplot(122, projection='3d')
                    p2 = ax2.scatter(x_coord, y_coord, z_coord, s=2, cmap="jet", c=s_l, label="labeled " + namestring)
                    p2.set_clim(vmin, vmax)
                    fig.colorbar(p2, fraction=0.045, pad=0.05)
                    ax2.set_xlabel('X', fontsize=10)
                    ax2.set_ylabel('Y', fontsize=10)
                    ax2.set_zlabel('Z', fontsize=10)
                    plt.title("Labeled Stress " + namestring + " of T: %d deg, P: %d kPa, D: %.2f mm" % (
                    int(input_names[1]), int(input_names[2]), float(input_names[3])))
                    plt.legend()
                    plt.show()

                    ratio = ratio_vect[1]
                    s = [s / ratio for s in s22]
                    s_l = [s / ratio for s in s22_l]
                    namestring = "s22"

                    x_coord = [coord_vect[3 * i] for i in range(coord_vect.shape[0] // 3)]
                    y_coord = [coord_vect[3 * i + 1] for i in range(coord_vect.shape[0] // 3)]
                    z_coord = [coord_vect[3 * i + 2] for i in range(coord_vect.shape[0] // 3)]

                    vmin = min(s_l + s)
                    vmax = max(s_l + s)
                    fig = plt.figure(dpi=128, figsize=(16, 8))
                    ax = fig.add_subplot(121, projection='3d')
                    p1 = ax.scatter(x_coord, y_coord, z_coord, s=2, cmap="jet", c=s, label="decoded " + namestring)
                    p1.set_clim(vmin, vmax)
                    fig.colorbar(p1, fraction=0.045, pad=0.05)
                    ax.set_xlabel('X', fontsize=10)
                    ax.set_ylabel('Y', fontsize=10)
                    ax.set_zlabel('Z', fontsize=10)
                    plt.title("Predicted Stress " + namestring + " of T: %d deg, P: %d kPa, D: %.2f mm" % (
                    int(input_names[1]), int(input_names[2]), float(input_names[3])))
                    plt.legend()

                    ax2 = fig.add_subplot(122, projection='3d')
                    p2 = ax2.scatter(x_coord, y_coord, z_coord, s=2, cmap="jet", c=s_l, label="labeled " + namestring)
                    p2.set_clim(vmin, vmax)
                    fig.colorbar(p2, fraction=0.045, pad=0.05)
                    ax2.set_xlabel('X', fontsize=10)
                    ax2.set_ylabel('Y', fontsize=10)
                    ax2.set_zlabel('Z', fontsize=10)
                    plt.title("Labeled Stress " + namestring + " of T: %d deg, P: %d kPa, D: %.2f mm" % (
                    int(input_names[1]), int(input_names[2]), float(input_names[3])))
                    plt.legend()
                    plt.show()

                    ratio = ratio_vect[2]
                    s = [s / ratio for s in s33]
                    s_l = [s / ratio for s in s33_l]
                    namestring = "s33"

                    x_coord = [coord_vect[3 * i] for i in range(coord_vect.shape[0] // 3)]
                    y_coord = [coord_vect[3 * i + 1] for i in range(coord_vect.shape[0] // 3)]
                    z_coord = [coord_vect[3 * i + 2] for i in range(coord_vect.shape[0] // 3)]

                    vmin = min(s_l + s)
                    vmax = max(s_l + s)
                    fig = plt.figure(dpi=128, figsize=(16, 8))
                    ax = fig.add_subplot(121, projection='3d')
                    p1 = ax.scatter(x_coord, y_coord, z_coord, s=2, cmap="jet", c=s, label="decoded " + namestring)
                    p1.set_clim(vmin, vmax)
                    fig.colorbar(p1, fraction=0.045, pad=0.05)
                    ax.set_xlabel('X', fontsize=10)
                    ax.set_ylabel('Y', fontsize=10)
                    ax.set_zlabel('Z', fontsize=10)
                    plt.title("Predicted Stress " + namestring + " of T: %d deg, P: %d kPa, D: %.2f mm" % (
                    int(input_names[1]), int(input_names[2]), float(input_names[3])))
                    plt.legend()

                    ax2 = fig.add_subplot(122, projection='3d')
                    p2 = ax2.scatter(x_coord, y_coord, z_coord, s=2, cmap="jet", c=s_l, label="labeled " + namestring)
                    p2.set_clim(vmin, vmax)
                    fig.colorbar(p2, fraction=0.045, pad=0.05)
                    ax2.set_xlabel('X', fontsize=10)
                    ax2.set_ylabel('Y', fontsize=10)
                    ax2.set_zlabel('Z', fontsize=10)
                    plt.title("Labeled Stress " + namestring + " of T: %d deg, P: %d kPa, D: %.2f mm" % (
                    int(input_names[1]), int(input_names[2]), float(input_names[3])))
                    plt.legend()
                    plt.show()

                if save and epoch > epoch_n-100:
                    vect_outputs = recon[0, :].detach().cpu().numpy()
                    temp_name = savedir + "CODER_" + file_name[0].split("\\")[-1][:-4]+".csv"
                    np.savetxt(temp_name, vect_outputs.T, delimiter=",")


            valid_loss_mean = np.mean(np.array(loss_rec))
            results_rec.append([lr, train_loss_mean, valid_loss_mean])

            if valid_loss_mean < min(loss_rec_all_valid):
                torch.save(autocoder.state_dict(), "./params1-noshuffle/" + type + "_coder_best.pth")
                np.savetxt("./params1-noshuffle/" + type + "_training_rec.txt", np.array(results_rec))
            loss_rec_all_valid.append(valid_loss_mean)
            print("Epoch %d, Train Loss: %.5f, Valid Loss: %.5f"%(epoch, train_loss_mean, valid_loss_mean))
            torch.save(autocoder.state_dict(), "./params1-noshuffle/" + type + "_coder_last.pth")


def train_predictor(type="BM", save=False, display=False, step=10):
    trainset_dir = r"E:\Data\DATASET\SealDigitTwin\STRESS\TRAIN\\"
    validset_dir = r"E:\Data\DATASET\SealDigitTwin\STRESS\VALID\\"

    ratio_train = True
    if ratio_train:
        ratio_vect = [4.5, 1.0, 4.5, 12.0, 25.0, 8.0]
    else:
        ratio_vect = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    train_dl = DataLoader(PreDataloader(trainset_dir, type=type, ratio_vect=ratio_vect),
                          batch_size=1,
                          shuffle=True)
    valid_dl = DataLoader(PreDataloader(validset_dir, type=type, ratio_vect=ratio_vect),
                          batch_size=1,
                          shuffle=True)
    lr = 1e-4
    if type == "BM":
        vect_length = 15184
    elif type == "NB":
        vect_length = 13120
    else:
        vect_length = 26016

    if save:
        savedir = "I:\\DigitRubber_Dataset\\NPY\\VALID_RESULTS\\COORD\\PREDICTOR\\"


    Code_predictor = Predictor(code_length=72, params_length=3, hiden_length=72).cuda()
    autocoder = VAE(vect_length=vect_length, code_length=72, num_layer=2, in_channel=6).cuda()

    checkpoint = torch.load("./params1-noshuffle/" + type + "_coder_best.pth")
    autocoder.load_state_dict(checkpoint)
    autocoder.eval()

    # checkpoint = torch.load(type + "_Predictor_best.pth")
    # Code_predictor.load_state_dict(checkpoint)
    if os.path.exists("./params1-noshuffle/" + type + "_Predictor_last.pth"):
        checkpoint = torch.load("./params1-noshuffle/" + type + "_Predictor_best.pth")
        Code_predictor.load_state_dict(checkpoint)
        results_rec = np.loadtxt("./params1-noshuffle/" + type + "_predictor_training_rec.txt")
        loss_rec_all_valid = results_rec[:, 1].tolist()
        loss_rec_all_train = results_rec[:, 2].tolist()
        results_rec = results_rec.tolist()
    else:
        loss_rec_all_valid = [10000]
        loss_rec_all_train = [10000]
        results_rec = []

    optimizer = torch.optim.Adam(Code_predictor.parameters(), lr=lr)
    epoch_n = 600
    loss_fn = torch.nn.MSELoss(reduce=True, reduction='sum')

    loss_rec = []
    for i in range(epoch_n - len(results_rec) * step):
        if i == 0:
            epoch = len(results_rec) * step
        epoch += 1
        Code_predictor.train()

        if epoch % 200 == 199:
            lr = lr / 2
            optimizer = torch.optim.Adam(Code_predictor.parameters(), lr=lr)
        # train_data = []
        loss_rec = []
        for i, [inputs, vects, _] in enumerate(train_dl):
            codes, _ = autocoder.encode(vects)
            outputs = Code_predictor(inputs).unsqueeze(1)
            # predicted_stress = autocoder.decode(outputs)
            Code_predictor.zero_grad()
            loss = loss_fn(codes, outputs)
            loss.backward()
            optimizer.step()
            loss_rec.append(loss.item())
        # train_data = np.array(train_data)
        # np.savetxt(type+"_train_codes_PRE_ADD.txt", train_data)

        # valid_data = []
        if epoch % step == 0 or epoch == epoch_n - 1:
            train_loss_mean = np.mean(np.array(loss_rec))
            loss_rec = []
            for i, [inputs, vects, file_name] in enumerate(valid_dl):
                codes, _ = autocoder.encode(vects)
                outputs = Code_predictor(inputs).unsqueeze(1)
                loss = loss_fn(codes, outputs)
                loss_rec.append(loss.item())

                '''下面的代码是用于可视化的，训练的时候可以注释掉'''
                if display:
                    recon = autocoder.decode(outputs)
                    vect_outputs = recon[0, :].detach().cpu().numpy()
                    vect_init = vects[0, :].detach().cpu().numpy()
                    input_names = file_name[0].split("\\")[-1].split("_")

                    s11 = vect_outputs[0, :].tolist()
                    s22 = vect_outputs[1, :].tolist()
                    s33 = vect_outputs[2, :].tolist()
                    s12 = vect_outputs[3, :].tolist()
                    s13 = vect_outputs[4, :].tolist()
                    s23 = vect_outputs[5, :].tolist()

                    s11_l = vect_init[0, :].tolist()
                    s22_l = vect_init[1, :].tolist()
                    s33_l = vect_init[2, :].tolist()
                    s12_l = vect_init[3, :].tolist()
                    s13_l = vect_init[4, :].tolist()
                    s23_l = vect_init[5, :].tolist()

                    temp_coord_filename = "E:\Data\DATASET\SealDigitTwin\\\COORD\\TRAIN\\" + file_name[0].split("\\")[
                                                                                                 -1][
                                                                                             :-10] + "COORD.npy"
                    try:
                        coord_vect = np.load(temp_coord_filename)
                    except:
                        temp_coord_filename = "E:\Data\DATASET\SealDigitTwin\\\COORD\\VALID\\" + \
                                              file_name[0].split("\\")[
                                                  -1][
                                              :-10] + "COORD.npy"
                        coord_vect = np.load(temp_coord_filename)

                    x_coord = [coord_vect[3 * i] for i in range(coord_vect.shape[0] // 3)]
                    y_coord = [coord_vect[3 * i + 1] for i in range(coord_vect.shape[0] // 3)]
                    z_coord = [coord_vect[3 * i + 2] for i in range(coord_vect.shape[0] // 3)]

                    #
                    ratio = ratio_vect[0]
                    s = [s / ratio for s in s11]
                    s_l = [s / ratio for s in s11_l]
                    namestring = "s11"

                    vmin = min(s_l + s)
                    vmax = max(s_l + s)
                    fig = plt.figure(dpi=128, figsize=(16, 8))
                    ax = fig.add_subplot(121, projection='3d')
                    p1 = ax.scatter(x_coord, y_coord, z_coord, s=2, cmap="jet", c=s, label="decoded " + namestring)
                    p1.set_clim(vmin, vmax)
                    fig.colorbar(p1, fraction=0.045, pad=0.05)
                    ax.set_xlabel('X', fontsize=10)
                    ax.set_ylabel('Y', fontsize=10)
                    ax.set_zlabel('Z', fontsize=10)
                    plt.title("Predicted Stress " + namestring + " of T: %d deg, P: %d kPa, D: %.2f mm" % (
                        int(input_names[1]), int(input_names[2]), float(input_names[3])))
                    plt.legend()

                    ax2 = fig.add_subplot(122, projection='3d')
                    p2 = ax2.scatter(x_coord, y_coord, z_coord, s=2, cmap="jet", c=s_l, label="labeled " + namestring)
                    p2.set_clim(vmin, vmax)
                    fig.colorbar(p2, fraction=0.045, pad=0.05)
                    ax2.set_xlabel('X', fontsize=10)
                    ax2.set_ylabel('Y', fontsize=10)
                    ax2.set_zlabel('Z', fontsize=10)
                    plt.title("Labeled Stress " + namestring + " of T: %d deg, P: %d kPa, D: %.2f mm" % (
                        int(input_names[1]), int(input_names[2]), float(input_names[3])))
                    plt.legend()
                    plt.show()

                    ratio = ratio_vect[1]
                    s = [s / ratio for s in s22]
                    s_l = [s / ratio for s in s22_l]
                    namestring = "s22"

                    x_coord = [coord_vect[3 * i] for i in range(coord_vect.shape[0] // 3)]
                    y_coord = [coord_vect[3 * i + 1] for i in range(coord_vect.shape[0] // 3)]
                    z_coord = [coord_vect[3 * i + 2] for i in range(coord_vect.shape[0] // 3)]

                    vmin = min(s_l + s)
                    vmax = max(s_l + s)
                    fig = plt.figure(dpi=128, figsize=(16, 8))
                    ax = fig.add_subplot(121, projection='3d')
                    p1 = ax.scatter(x_coord, y_coord, z_coord, s=2, cmap="jet", c=s, label="decoded " + namestring)
                    p1.set_clim(vmin, vmax)
                    fig.colorbar(p1, fraction=0.045, pad=0.05)
                    ax.set_xlabel('X', fontsize=10)
                    ax.set_ylabel('Y', fontsize=10)
                    ax.set_zlabel('Z', fontsize=10)
                    plt.title("Predicted Stress " + namestring + " of T: %d deg, P: %d kPa, D: %.2f mm" % (
                        int(input_names[1]), int(input_names[2]), float(input_names[3])))
                    plt.legend()

                    ax2 = fig.add_subplot(122, projection='3d')
                    p2 = ax2.scatter(x_coord, y_coord, z_coord, s=2, cmap="jet", c=s_l, label="labeled " + namestring)
                    p2.set_clim(vmin, vmax)
                    fig.colorbar(p2, fraction=0.045, pad=0.05)
                    ax2.set_xlabel('X', fontsize=10)
                    ax2.set_ylabel('Y', fontsize=10)
                    ax2.set_zlabel('Z', fontsize=10)
                    plt.title("Labeled Stress " + namestring + " of T: %d deg, P: %d kPa, D: %.2f mm" % (
                        int(input_names[1]), int(input_names[2]), float(input_names[3])))
                    plt.legend()
                    plt.show()

                    ratio = ratio_vect[2]
                    s = [s / ratio for s in s33]
                    s_l = [s / ratio for s in s33_l]
                    namestring = "s33"

                    x_coord = [coord_vect[3 * i] for i in range(coord_vect.shape[0] // 3)]
                    y_coord = [coord_vect[3 * i + 1] for i in range(coord_vect.shape[0] // 3)]
                    z_coord = [coord_vect[3 * i + 2] for i in range(coord_vect.shape[0] // 3)]

                    vmin = min(s_l + s)
                    vmax = max(s_l + s)
                    fig = plt.figure(dpi=128, figsize=(16, 8))
                    ax = fig.add_subplot(121, projection='3d')
                    p1 = ax.scatter(x_coord, y_coord, z_coord, s=2, cmap="jet", c=s, label="decoded " + namestring)
                    p1.set_clim(vmin, vmax)
                    fig.colorbar(p1, fraction=0.045, pad=0.05)
                    ax.set_xlabel('X', fontsize=10)
                    ax.set_ylabel('Y', fontsize=10)
                    ax.set_zlabel('Z', fontsize=10)
                    plt.title("Predicted Stress " + namestring + " of T: %d deg, P: %d kPa, D: %.2f mm" % (
                        int(input_names[1]), int(input_names[2]), float(input_names[3])))
                    plt.legend()

                    ax2 = fig.add_subplot(122, projection='3d')
                    p2 = ax2.scatter(x_coord, y_coord, z_coord, s=2, cmap="jet", c=s_l, label="labeled " + namestring)
                    p2.set_clim(vmin, vmax)
                    fig.colorbar(p2, fraction=0.045, pad=0.05)
                    ax2.set_xlabel('X', fontsize=10)
                    ax2.set_ylabel('Y', fontsize=10)
                    ax2.set_zlabel('Z', fontsize=10)
                    plt.title("Labeled Stress " + namestring + " of T: %d deg, P: %d kPa, D: %.2f mm" % (
                        int(input_names[1]), int(input_names[2]), float(input_names[3])))
                    plt.legend()
                    plt.show()
                # if save and epoch > epoch_n-100:
                #     vect_outputs = shape_outputs[0, :].detach().cpu().numpy()
                #     temp_name = savedir + "PREDICTOR_" + filename[0].split("\\")[-1][:-4] + ".csv"
                #     np.savetxt(temp_name, vect_outputs.T, delimiter=",")
            valid_loss_mean = np.mean(np.array(loss_rec))
            results_rec.append([lr, train_loss_mean, valid_loss_mean])

            if valid_loss_mean < min(loss_rec_all_valid):
                torch.save(Code_predictor.state_dict(), "./params1-noshuffle/" + type + "_predictor_best.pth")
                np.savetxt("./params1-noshuffle/" + type + "_predictor_training_rec.txt", np.array(results_rec))
            loss_rec_all_train.append(train_loss_mean)
            loss_rec_all_valid.append(valid_loss_mean)
            print("Epoch %d, Train Loss: %.5f, Valid Loss: %.5f" % (epoch, train_loss_mean, valid_loss_mean))
            torch.save(Code_predictor.state_dict(), "./params1-noshuffle/" + type + "_predictor_last.pth")
            # valid_data = np.array(valid_data)
            # np.savetxt(type+"_valid_codes_PRE.txt", valid_data)

            # valid_loss_mean = np.mean(np.array(loss_rec))
            # if valid_loss_mean < min(loss_rec_all):
            #     torch.save(Code_predictor.state_dict(), type + "_Predictor_best.pth")
            # loss_rec_all.append(valid_loss_mean)
            # print("Epoch %d, Train Loss: %.4f, Valid Loss: %.4f" % (epoch, train_loss_mean, valid_loss_mean))
            # torch.save(Code_predictor.state_dict(), type + "_Predictor_last.pth")


def eval_model(list_load, type, save=True):
    if type == "BM":
        vect_length = 15184
    elif type == "NB":
        vect_length = 13120
    else:
        vect_length = 26016
    if save:
        savedir = "I:\\DigitRubber_Dataset\\NPY\\VALID_RESULTS\\COORD\\PREDICTED\\"

    Code_predictor = Predictor(code_length=24, params_length=3, hiden_length=64).cuda()
    autocoder = AUTOCODER_array(vect_length=vect_length, code_length=24, num_layer=3).cuda()

    checkpoint = torch.load(type + "_coder_best.pth")
    autocoder.load_state_dict(checkpoint)
    autocoder.eval()

    checkpoint = torch.load(type + "_Predictor_best.pth")
    Code_predictor.load_state_dict(checkpoint)
    Code_predictor.eval()

    for i in range(len(list_load)):
        temp_input = torch.from_numpy(np.array(list_load[i], dtype="float32")).unsqueeze(0).cuda(0)
        outputs = Code_predictor(temp_input)
        # print(inputs)
        shape_outputs = autocoder(outputs, coder=1)
        vect_outputs = shape_outputs[0, :].detach().cpu().numpy().T
        temp_name = savedir + "PREDICTED_" + str(i) + "_" + type+ "_"+str(list_load[i][0]) +"_"+ str(list_load[i][1])+"_"+ str(list_load[i][2])+"_COORD.csv"
        np.savetxt(temp_name, vect_outputs, delimiter=",")


if __name__ == '__main__':
    train_autocoder(type="BM", save=False, display=False, step=5)
    # train_autocoder(type="PR", save=False, display=False, step=10)
    # train_autocoder(type="NB", save=False, display=False, step=10)

    train_predictor(type="BM", save=False, display=False, step=10)
    # train_predictor(type="PR", save=False, display=False, step=10)
    # train_predictor(type="NB", save=False, display=False, step=10)