import torch
import numpy as np
from STRESS.network import blocked_AUTOCODER, Predictor, blocked_AUTOCODER_pr
from torch.utils.data import Dataset, DataLoader
import os
import time
import matplotlib.pyplot as plt
import cv2
from COORD_NL.train_coord import coord_predictor


class STRESS_dataloader(Dataset):
    def __init__(self, file_dir, type="BM", ratio_vect=[4.0, 1.0, 4.0, 10.0, 20.0, 8.0]):
        self.all_files = os.listdir(file_dir)
        self.all_inputs = [os.path.join(file_dir, file) for file in self.all_files if (file.endswith("STRESS.npy") and file.startswith(type))]
        self.type = type
        self.ratio_vect = torch.from_numpy(np.array(ratio_vect, dtype="float32")).cuda()

    def __len__(self):
        return len(self.all_inputs)

    def __getitem__(self, item):
        filename = self.all_inputs[item]
        vect = np.load(filename)
        vect_torch = torch.from_numpy(vect).cuda()
        vect_torch_m = torch.transpose(torch.transpose(vect_torch, 0, 1) * self.ratio_vect, 0, 1)
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


def train_autocoder(type="BM", save=True, display=False, step=10):
    trainset_dir = "I:\\DigitRubber_Dataset\\NPY\\STRESS\\TRAIN\\"
    validset_dir = "I:\\DigitRubber_Dataset\\NPY\\STRESS\\VALID\\"
    ratio_train = True
    if type == "BM":
        vect_length = 15184
    elif type == "NB":
        vect_length = 13120
    else:
        vect_length = 26016
    if save:
        savedir = "I:\\DigitRubber_Dataset\\NPY\\VALID_RESULTS\\STRESS\\CODER\\"

    if ratio_train:
        ratio_vect = [4.5, 1.0, 4.5, 12.0, 25.0, 8.0]
    else:
        ratio_vect = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    train_dl = DataLoader(STRESS_dataloader(trainset_dir, type=type, ratio_vect=ratio_vect),
                          batch_size=1,
                          shuffle=True)
    valid_dl = DataLoader(STRESS_dataloader(validset_dir, type=type, ratio_vect=ratio_vect),
                          batch_size=1,
                          shuffle=True)
    lr = 1e-4
    autocoder = blocked_AUTOCODER(vect_length=vect_length, code_length=72, num_layer=3, in_channel=6).cuda()
    if type=="PR":
        autocoder = blocked_AUTOCODER_pr(vect_length=vect_length, code_length=72, num_layer=3, in_channel=6).cuda()
    # checkpoint = torch.load(type + "_coder_best.pth")
    autocoder.load_state_dict(torch.load(type + "_coder_best.pth"))
    loss_rec_all_valid = np.loadtxt(type + "_loss_valid.txt").tolist()
    loss_rec_all_train = np.loadtxt(type + "_loss_train.txt").tolist()
    # loss_rec_all_valid = [1]
    # loss_rec_all_train = []

    optimizer = torch.optim.Adam(autocoder.parameters(), lr=lr)
    epoch_n = 2
    loss_fn = torch.nn.MSELoss(reduce=True, reduction='mean')

    loss_rec = []
    for epoch in range(epoch_n):
        epoch += 1
        autocoder.train()

        for i, (vects, _) in enumerate(train_dl):
            outputs = autocoder(vects)                  # Stress -> Stress Code -> Stress
            autocoder.zero_grad()
            loss = loss_fn(vects, outputs)
            loss.backward()
            optimizer.step()
            loss_rec.append(loss.item())

        if epoch % step == 0 or epoch == epoch_n - 1:
            train_loss_mean = np.mean(np.array(loss_rec))
            loss_rec_all_train.append(train_loss_mean)
            loss_rec = []
            for i, (vects, file_name) in enumerate(valid_dl):
                outputs = autocoder(vects.clone())
                loss = loss_fn(vects, outputs)
                loss_rec.append(loss.item())
                '''for visualization'''
                if display:
                    vect_outputs = outputs[0, :].detach().cpu().numpy()
                    vect_init = vects[0, :].detach().cpu().numpy()

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
                    #
                    ratio = ratio_vect[3]
                    s = [s / ratio for s in s12]
                    s_l = [s / ratio for s in s12_l]
                    namestring = "s12"

                    temp_coord_filename = "I:\\DigitRubber_Dataset\\NPY\\COORD\\TRAIN\\" + file_name[0].split("\\")[-1][:-10] + "COORD.npy"
                    try:
                        coord_vect = np.load(temp_coord_filename)
                    except:
                        temp_coord_filename = "I:\\DigitRubber_Dataset\\NPY\\COORD\\VALID\\" + file_name[0].split("\\")[-1][:-10] + "COORD.npy"
                        coord_vect = np.load(temp_coord_filename)
                    x_coord = [coord_vect[3 * i] for i in range(coord_vect.shape[0] // 3)]
                    y_coord = [coord_vect[3 * i + 1] for i in range(coord_vect.shape[0] // 3)]
                    z_coord = [coord_vect[3 * i + 2] for i in range(coord_vect.shape[0] // 3)]
                    import matplotlib.pyplot as plt
                    vmin = min(s_l+s)
                    vmax = max(s_l+s)
                    fig = plt.figure(dpi=128, figsize=(16, 8))
                    ax = fig.add_subplot(121, projection='3d')
                    p1 = ax.scatter(x_coord, y_coord, z_coord, s=2, cmap="jet", c=s, label="decoded "+namestring)
                    p1.set_clim(vmin, vmax)
                    fig.colorbar(p1, fraction=0.045, pad=0.05)
                    ax.set_xlabel('X', fontsize=10)
                    ax.set_ylabel('Y', fontsize=10)
                    ax.set_zlabel('Z', fontsize=10)
                    plt.title("Decoded "+namestring+" Stress Distribution")
                    plt.legend()

                    ax2 = fig.add_subplot(122, projection='3d')
                    p2 = ax2.scatter(x_coord, y_coord, z_coord, s=2, cmap="jet", c=s_l, label="labeled "+namestring)
                    p2.set_clim(vmin, vmax)
                    fig.colorbar(p2, fraction=0.045, pad=0.05)
                    ax2.set_xlabel('X', fontsize=10)
                    ax2.set_ylabel('Y', fontsize=10)
                    ax2.set_zlabel('Z', fontsize=10)
                    plt.title("Label "+namestring+" Stress Distribution")
                    plt.legend()
                    plt.show()

                    ratio = ratio_vect[4]
                    s = [s / ratio for s in s13]
                    s_l = [s / ratio for s in s13_l]
                    namestring = "s13"

                    temp_coord_filename = "I:\\DigitRubber_Dataset\\NPY\\COORD\\TRAIN\\" + file_name[0].split("\\")[-1][
                                                                                           :-10] + "COORD.npy"
                    try:
                        coord_vect = np.load(temp_coord_filename)
                    except:
                        temp_coord_filename = "I:\\DigitRubber_Dataset\\NPY\\COORD\\VALID\\" + file_name[0].split("\\")[-1][
                                                                                               :-10] + "COORD.npy"
                        coord_vect = np.load(temp_coord_filename)
                    x_coord = [coord_vect[3 * i] for i in range(coord_vect.shape[0] // 3)]
                    y_coord = [coord_vect[3 * i + 1] for i in range(coord_vect.shape[0] // 3)]
                    z_coord = [coord_vect[3 * i + 2] for i in range(coord_vect.shape[0] // 3)]
                    import matplotlib.pyplot as plt
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
                    plt.title("Decoded " + namestring + " Stress Distribution")
                    plt.legend()

                    ax2 = fig.add_subplot(122, projection='3d')
                    p2 = ax2.scatter(x_coord, y_coord, z_coord, s=2, cmap="jet", c=s_l, label="labeled " + namestring)
                    p2.set_clim(vmin, vmax)
                    fig.colorbar(p2, fraction=0.045, pad=0.05)
                    ax2.set_xlabel('X', fontsize=10)
                    ax2.set_ylabel('Y', fontsize=10)
                    ax2.set_zlabel('Z', fontsize=10)
                    plt.title("Label " + namestring + " Stress Distribution")
                    plt.legend()
                    plt.show()

                    ratio = ratio_vect[5]
                    s = [s / ratio for s in s23]
                    s_l = [s / ratio for s in s23_l]
                    namestring = "s23"

                    temp_coord_filename = "I:\\DigitRubber_Dataset\\NPY\\COORD\\TRAIN\\" + file_name[0].split("\\")[-1][
                                                                                           :-10] + "COORD.npy"
                    try:
                        coord_vect = np.load(temp_coord_filename)
                    except:
                        temp_coord_filename = "I:\\DigitRubber_Dataset\\NPY\\COORD\\VALID\\" + file_name[0].split("\\")[-1][
                                                                                               :-10] + "COORD.npy"
                        coord_vect = np.load(temp_coord_filename)
                    x_coord = [coord_vect[3 * i] for i in range(coord_vect.shape[0] // 3)]
                    y_coord = [coord_vect[3 * i + 1] for i in range(coord_vect.shape[0] // 3)]
                    z_coord = [coord_vect[3 * i + 2] for i in range(coord_vect.shape[0] // 3)]
                    import matplotlib.pyplot as plt
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
                    plt.title("Decoded " + namestring + " Stress Distribution")
                    plt.legend()

                    ax2 = fig.add_subplot(122, projection='3d')
                    p2 = ax2.scatter(x_coord, y_coord, z_coord, s=2, cmap="jet", c=s_l, label="labeled " + namestring)
                    p2.set_clim(vmin, vmax)
                    fig.colorbar(p2, fraction=0.045, pad=0.05)
                    ax2.set_xlabel('X', fontsize=10)
                    ax2.set_ylabel('Y', fontsize=10)
                    ax2.set_zlabel('Z', fontsize=10)
                    plt.title("Label " + namestring + " Stress Distribution")
                    plt.legend()
                    plt.show()

                if save and epoch > epoch_n - 100:
                    vect_outputs = outputs[0, :].detach().cpu().numpy().T / np.array(ratio_vect)
                    temp_name = savedir + "CODER_" + file_name[0].split("\\")[-1][:-4] + ".csv"
                    np.savetxt(temp_name, vect_outputs, delimiter=",")

            valid_loss_mean = np.mean(np.array(loss_rec))

            if valid_loss_mean < min(loss_rec_all_valid):
                torch.save(autocoder.state_dict(), type + "_coder_best.pth")
                np.savetxt(type + "_loss_valid.txt", np.array(loss_rec_all_valid))
                np.savetxt(type + "_loss_train.txt", np.array(loss_rec_all_train))
            loss_rec_all_valid.append(valid_loss_mean)
            print("Epoch %d, Train Loss: %.5f, Valid Loss: %.5f"%(epoch, train_loss_mean, valid_loss_mean))
            torch.save(autocoder.state_dict(), type + "_coder_last.pth")


def train_predictor(type="BM", save=True, display=False, step=10):
    trainset_dir = "I:\\DigitRubber_Dataset\\NPY\\STRESS\\TRAIN\\"
    validset_dir = "I:\\DigitRubber_Dataset\\NPY\\STRESS\\VALID\\"
    ratio_train = True
    if type == "BM":
        vect_length = 15184
    elif type == "NB":
        vect_length = 13120
    else:
        vect_length = 26016
    if save:
        savedir = "I:\\DigitRubber_Dataset\\NPY\\VALID_RESULTS\\STRESS\\PREDICTOR\\"
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
    Code_predictor = Predictor(code_length=72, params_length=3, hiden_length=72).cuda()

    autocoder = blocked_AUTOCODER(vect_length=vect_length, code_length=72, num_layer=3, in_channel=6).cuda()
    if type=="PR":
        autocoder = blocked_AUTOCODER_pr(vect_length=vect_length, code_length=72, num_layer=3, in_channel=6).cuda()

    checkpoint = torch.load(type + "_coder_best.pth")
    autocoder.load_state_dict(checkpoint)
    autocoder.eval()

    checkpoint = torch.load(type + "_Predictor_last.pth")
    Code_predictor.load_state_dict(checkpoint)
    optimizer = torch.optim.Adam(Code_predictor.parameters(), lr=lr)
    epoch_n = 2
    loss_fn = torch.nn.MSELoss(reduce=True, reduction='mean')

    loss_rec_all = [10]

    loss_rec = []
    for epoch in range(epoch_n):
        epoch += 1
        Code_predictor.train()

        for i, [inputs, vects, _] in enumerate(train_dl):
            codes = autocoder(vects, coder=0)               # Aimed code to be fed into Decoder
            outputs = Code_predictor(inputs).unsqueeze(1)   # code predicted from Predictor
            Code_predictor.zero_grad()
            loss = loss_fn(codes, outputs)
            loss.backward()
            optimizer.step()

            loss_rec.append(loss.item())


        if epoch % step == 0 or epoch == epoch_n - 1:
            train_loss_mean = np.mean(np.array(loss_rec))
            loss_rec = []
            for i, [inputs, vects, file_name] in enumerate(valid_dl):
                codes = autocoder(vects, coder=0)
                outputs = Code_predictor(inputs).unsqueeze(1)
                # print(inputs)
                stress_outputs = autocoder(outputs, coder=1)
                loss = loss_fn(codes, outputs)
                loss_rec.append(loss.item())

                '''for visualization'''
                if display:
                    vect_outputs = stress_outputs[0, :].detach().cpu().numpy()
                    vect_init = vects[0, :].detach().cpu().numpy()
                    inputs_num = inputs[0, :].detach().cpu().numpy()

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

                    temp_coord_filename = "I:\\DigitRubber_Dataset\\NPY\\COORD\\TRAIN\\" + file_name[0].split("\\")[-1][:-10] + "COORD.npy"
                    try:
                        coord_vect = np.load(temp_coord_filename)
                    except:
                        temp_coord_filename = "I:\\DigitRubber_Dataset\\NPY\\COORD\\VALID\\" + file_name[0].split("\\")[-1][:-10] + "COORD.npy"
                        coord_vect = np.load(temp_coord_filename)

                    x_coord = [coord_vect[3 * i] for i in range(coord_vect.shape[0] // 3)]
                    y_coord = [coord_vect[3 * i + 1] for i in range(coord_vect.shape[0] // 3)]
                    z_coord = [coord_vect[3 * i + 2] for i in range(coord_vect.shape[0] // 3)]

                    import matplotlib.pyplot as plt

                    #
                    ratio = ratio_vect[0]
                    s = [s / ratio for s in s11]
                    s_l = [s / ratio for s in s11_l]
                    namestring = "s11"

                    vmin = min(s_l+s)
                    vmax = max(s_l+s)
                    fig = plt.figure(dpi=128, figsize=(16, 8))
                    ax = fig.add_subplot(121, projection='3d')
                    p1 = ax.scatter(x_coord, y_coord, z_coord, s=2, cmap="jet", c=s, label="decoded "+namestring)
                    p1.set_clim(vmin, vmax)
                    fig.colorbar(p1, fraction=0.045, pad=0.05)
                    ax.set_xlabel('X', fontsize=10)
                    ax.set_ylabel('Y', fontsize=10)
                    ax.set_zlabel('Z', fontsize=10)
                    plt.title("Predicted Stress "+namestring+" of T: %d deg, P: %d kPa, D: %.2f mm"%(int(inputs_num[0]), int(inputs_num[1]), inputs_num[2]))
                    plt.legend()

                    ax2 = fig.add_subplot(122, projection='3d')
                    p2 = ax2.scatter(x_coord, y_coord, z_coord, s=2, cmap="jet", c=s_l, label="labeled "+namestring)
                    p2.set_clim(vmin, vmax)
                    fig.colorbar(p2, fraction=0.045, pad=0.05)
                    ax2.set_xlabel('X', fontsize=10)
                    ax2.set_ylabel('Y', fontsize=10)
                    ax2.set_zlabel('Z', fontsize=10)
                    plt.title("Labeled Stress "+namestring+" of T: %d deg, P: %d kPa, D: %.2f mm"%(int(inputs_num[0]), int(inputs_num[1]), inputs_num[2]))
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
                    plt.title("Predicted Stress "+namestring+" of T: %d deg, P: %d kPa, D: %.2f mm"%(int(inputs_num[0]), int(inputs_num[1]), inputs_num[2]))
                    plt.legend()

                    ax2 = fig.add_subplot(122, projection='3d')
                    p2 = ax2.scatter(x_coord, y_coord, z_coord, s=2, cmap="jet", c=s_l, label="labeled " + namestring)
                    p2.set_clim(vmin, vmax)
                    fig.colorbar(p2, fraction=0.045, pad=0.05)
                    ax2.set_xlabel('X', fontsize=10)
                    ax2.set_ylabel('Y', fontsize=10)
                    ax2.set_zlabel('Z', fontsize=10)
                    plt.title("Labeled Stress "+namestring+" of T: %d deg, P: %d kPa, D: %.2f mm"%(int(inputs_num[0]), int(inputs_num[1]), inputs_num[2]))
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
                    plt.title("Predicted Stress "+namestring+" of T: %d deg, P: %d kPa, D: %.2f mm"%(int(inputs_num[0]), int(inputs_num[1]), inputs_num[2]))
                    plt.legend()

                    ax2 = fig.add_subplot(122, projection='3d')
                    p2 = ax2.scatter(x_coord, y_coord, z_coord, s=2, cmap="jet", c=s_l, label="labeled " + namestring)
                    p2.set_clim(vmin, vmax)
                    fig.colorbar(p2, fraction=0.045, pad=0.05)
                    ax2.set_xlabel('X', fontsize=10)
                    ax2.set_ylabel('Y', fontsize=10)
                    ax2.set_zlabel('Z', fontsize=10)
                    plt.title("Labeled Stress "+namestring+" of T: %d deg, P: %d kPa, D: %.2f mm"%(int(inputs_num[0]), int(inputs_num[1]), inputs_num[2]))
                    plt.legend()
                    plt.show()

                if save and epoch > epoch_n - 100:
                    vect_outputs = stress_outputs[0, :].detach().cpu().numpy().T / np.array(ratio_vect)
                    temp_name = savedir + "PREDICTOR_" + file_name[0].split("\\")[-1][:-4] + ".csv"
                    np.savetxt(temp_name, vect_outputs, delimiter=",")

            valid_loss_mean = np.mean(np.array(loss_rec))
            if valid_loss_mean < min(loss_rec_all):
                torch.save(Code_predictor.state_dict(), type + "_Predictor_best.pth")
            loss_rec_all.append(valid_loss_mean)
            print("Epoch %d, Train Loss: %.4f, Valid Loss: %.4f" % (epoch, train_loss_mean, valid_loss_mean))
            torch.save(Code_predictor.state_dict(), type + "_Predictor_last.pth")


'''To evaluation model'''
def eval_model(list_load, type, save=True):
    ratio_train = True
    if type == "BM":
        vect_length = 15184
    elif type == "NB":
        vect_length = 13120
    else:
        vect_length = 26016
    if save:
        savedir = "I:\\DigitRubber_Dataset\\NPY\\VALID_RESULTS\\STRESS\\PREDICTED\\"
    if ratio_train:
        ratio_vect = [4.5, 1.0, 4.5, 12.0, 25.0, 8.0]
    else:
        ratio_vect = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    Code_predictor = Predictor(code_length=72, params_length=3, hiden_length=72).cuda()
    checkpoint = torch.load(type + "_Predictor_best.pth")
    Code_predictor.load_state_dict(checkpoint)
    Code_predictor.eval()

    autocoder = blocked_AUTOCODER(vect_length=vect_length, code_length=72, num_layer=3, in_channel=6).cuda()
    if type == "PR":
        autocoder = blocked_AUTOCODER_pr(vect_length=vect_length, code_length=72, num_layer=3, in_channel=6).cuda()
    autocoder.load_state_dict(torch.load(type + "_coder_best.pth"))
    autocoder.eval()

    for i in range(len(list_load)):
        temp_input = torch.from_numpy(np.array(list_load[i], dtype="float32")).unsqueeze(0).cuda(0)
        outputs = Code_predictor(temp_input).unsqueeze(1)
        # print(inputs)
        stress_outputs = autocoder(outputs, coder=1)
        vect_outputs = stress_outputs[0, :].detach().cpu().numpy().T / np.array(ratio_vect)
        temp_name = savedir + "PREDICTED_" + str(i) + "_" + type+ "_"+str(list_load[i][0]) +"_"+ str(list_load[i][1])+"_"+ str(list_load[i][2])+".csv"
        np.savetxt(temp_name, vect_outputs, delimiter=",")


if __name__ == '__main__':
    train_autocoder(type="BM", save=True, display=False, step=10)
    train_predictor(type="BM", save=True, display=False, step=10)
    # # train_autocoder(type="NB", save=True, display=False, step=10)
    # train_predictor(type="NB", save=True, display=False, step=10)
    # train_autocoder(type="PR", save=True, display=False, step=10)
    # train_predictor(type="PR", save=True, display=False, step=10)
    # eval_model(list_load=arr.tolist(),
    #            type="BM")
    # eval_model(list_load=arr.tolist(),
    #            type="NB")
    # eval_model(list_load=arr.tolist(),
    #            type="PR")



    # arr = np.loadtxt(r"I:\DigitRubber_Dataset\NPY\VALID_RESULTS\\seriel3.txt", delimiter="\t", dtype="float32")
    # arr = arr[:, [1, 2, 0]]
    # eval_all_layers(arr.tolist())