import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from STRESS.network import VAE


class Code_dataloader(Dataset):
    def __init__(self, file_dir, type="BM"):
        self.all_files = os.listdir(file_dir)
        self.all_stress_inputs = [os.path.join(file_dir, file) for file in self.all_files if (file.endswith("STRESS.npy") and file.startswith(type))]
        self.all_coord_inputs = [os.path.join(file_dir, file) for file in self.all_files if (file.endswith("Coord.npy") and file.startswith(type))]
        self.type = type

    def __len__(self):
        return len(self.all_stress_inputs)

    def __getitem__(self, item):
        stress_filename = self.all_stress_inputs[item]
        coord_filename = self.all_coord_inputs[item]
        stress_vect = np.load(stress_filename)
        coord_vect = np.load(coord_filename)
        vect_torch_stress = torch.from_numpy(stress_vect).cuda()
        vect_torch_coord = torch.from_numpy(coord_vect).cuda()
        return vect_torch_coord, vect_torch_stress, stress_filename


class Continuum(nn.Module):
    def __init__(self, coord_code_length=24, stress_code_length=72, hiden_length=48):
        super(Continuum, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(coord_code_length, hiden_length),
            nn.ReLU(),
            nn.Linear(hiden_length, hiden_length),
            nn.ReLU(),
            nn.Linear(hiden_length, hiden_length),
            nn.ReLU(),
            nn.Linear(hiden_length, hiden_length),
            nn.ReLU(),
            nn.Linear(hiden_length, stress_code_length),
        )

    def forward(self, x):
        return self.model(x)


def train_continuum(type="BM"):
    step = 5
    display = True

    trainset_dir = r"E:\Data\DATASET\SealDigitTwin\CONTINUUM\TRAIN\\"
    validset_dir = r"E:\Data\DATASET\SealDigitTwin\CONTINUUM\VALID\\"

    train_dl = DataLoader(Code_dataloader(trainset_dir, type=type),
                          batch_size=1,
                          shuffle=True)
    valid_dl = DataLoader(Code_dataloader(validset_dir, type=type),
                          batch_size=1,
                          shuffle=True)

    Code_predictor = Continuum(coord_code_length=24, stress_code_length=72, hiden_length=48).cuda()

    if type == "BM":
        vect_length = 15184
    elif type == "NB":
        vect_length = 13120
    else:
        vect_length = 26016
    autocoder = VAE(vect_length=vect_length, code_length=72, num_layer=2, in_channel=6).cuda()
    checkpoint = torch.load("../STRESS//params1/" + type + "_coder_last.pth")
    autocoder.load_state_dict(checkpoint)
    autocoder.eval()

    if os.path.exists("./params/" + type + "_Continuum_last.pth"):
        checkpoint = torch.load("./params/" + type + "_Continuum_last.pth")
        Code_predictor.load_state_dict(checkpoint)
        results_rec = np.loadtxt("./params/" + type + "_Continuum_training_rec.txt")
        loss_rec_all_valid = results_rec[:, 2].tolist()
        loss_rec_all_train = results_rec[:, 1].tolist()
        results_rec = results_rec.tolist()
    else:
        loss_rec_all_valid = [10000]
        loss_rec_all_train = [10000]
        results_rec = []

    lr = 1e-5
    optimizer = torch.optim.Adam(Code_predictor.parameters(), lr=lr)
    epoch_n = 2210
    loss_fn = torch.nn.MSELoss(reduce=True, reduction='sum')

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
        for i, [coord_code, stress_code, _] in enumerate(train_dl):
            pre_stress_code = Code_predictor(coord_code)
            Code_predictor.zero_grad()
            recon = autocoder.decode(pre_stress_code.unsqueeze(0))
            label = autocoder.decode(stress_code)

            loss_1 = loss_fn(pre_stress_code, stress_code)
            loss_2 = loss_fn(recon, label)
            loss = loss_2 * 100 + loss_1
            loss.backward()
            optimizer.step()
            loss_rec.append(loss.item())

        # valid_data = []
        if epoch % step == 0 or epoch == epoch_n - 1:
            train_loss_mean = np.mean(np.array(loss_rec))
            loss_rec = []
            loss_2_rec = []
            for i, [coord_code, stress_code, file_name] in enumerate(valid_dl):
                pre_stress_code = Code_predictor(coord_code)
                loss_1 = loss_fn(pre_stress_code, stress_code)
                loss_2 = loss_fn(recon, label)
                loss = 100 * loss_2 + loss_1
                loss_rec.append(loss_2.item())
                loss_2_rec.append(loss_2.item())

                if display:
                    ratio_vect = [4.5, 1.0, 4.5, 12.0, 25.0, 8.0]
                    recon = autocoder.decode(pre_stress_code.unsqueeze(0))
                    label = autocoder.decode(stress_code)
                    vect_outputs = recon[0, :].detach().cpu().numpy()
                    vect_init = label[0, :].detach().cpu().numpy()
                    input_names = file_name[0].split("\\")[-1].split("_")

                    s11 = vect_outputs[0, :].tolist()
                    s22 = vect_outputs[1, :].tolist()
                    s33 = vect_outputs[2, :].tolist()

                    s11_l = vect_init[0, :].tolist()
                    s22_l = vect_init[1, :].tolist()
                    s33_l = vect_init[2, :].tolist()

                    temp_coord_filename = "E:\Data\DATASET\SealDigitTwin\\\COORD\\TRAIN\\" + file_name[0].split("\\")[-1][:-10] + "COORD.npy"
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
                    s_e = np.array(s) - np.array(s_l)
                    s_e.tolist()
                    namestring = "s11"

                    vmin = min(s_l + s)
                    vmax = max(s_l + s)
                    fig = plt.figure(dpi=128, figsize=(16, 8))
                    ax = fig.add_subplot(131, projection='3d')
                    p1 = ax.scatter(x_coord, y_coord, z_coord, s=2, cmap="jet", c=s, label="decoded " + namestring)
                    p1.set_clim(vmin, vmax)
                    fig.colorbar(p1, fraction=0.045, pad=0.05)
                    ax.set_xlabel('X', fontsize=10)
                    ax.set_ylabel('Y', fontsize=10)
                    ax.set_zlabel('Z', fontsize=10)
                    plt.title("Predicted Stress " + namestring + " of T: %d deg, P: %d kPa, D: %.2f mm" % (
                        int(input_names[1]), int(input_names[2]), float(input_names[3])))
                    plt.legend()

                    ax2 = fig.add_subplot(132, projection='3d')
                    p2 = ax2.scatter(x_coord, y_coord, z_coord, s=2, cmap="jet", c=s_l, label="labeled " + namestring)
                    p2.set_clim(vmin, vmax)
                    fig.colorbar(p2, fraction=0.045, pad=0.05)
                    ax2.set_xlabel('X', fontsize=10)
                    ax2.set_ylabel('Y', fontsize=10)
                    ax2.set_zlabel('Z', fontsize=10)
                    plt.title("Labeled Stress " + namestring + " of T: %d deg, P: %d kPa, D: %.2f mm" % (
                        int(input_names[1]), int(input_names[2]), float(input_names[3])))
                    plt.legend()

                    ax2 = fig.add_subplot(133, projection='3d')
                    p2 = ax2.scatter(x_coord, y_coord, z_coord, s=2, cmap="jet", c=s_e, label="labeled " + namestring)
                    # p2.set_clim(vmin, vmax)
                    fig.colorbar(p2, fraction=0.045, pad=0.05)
                    ax2.set_xlabel('X', fontsize=10)
                    ax2.set_ylabel('Y', fontsize=10)
                    ax2.set_zlabel('Z', fontsize=10)
                    plt.title("Labeled Stress " + namestring + " of T: %d deg, P: %d kPa, D: %.2f mm" % (
                        int(input_names[1]), int(input_names[2]), float(input_names[3])))
                    plt.legend()
                    plt.show()

            valid_loss_mean = np.mean(np.array(loss_rec))
            valid_loss_mean_2 = np.mean(np.array(loss_2_rec))
            results_rec.append([lr, train_loss_mean, valid_loss_mean])
            if valid_loss_mean < min(loss_rec_all_valid):
                torch.save(Code_predictor.state_dict(), "./params/" + type + "_Continuum_best.pth")
            np.savetxt("./params/" + type + "_Continuum_training_rec.txt", np.array(results_rec))
            loss_rec_all_train.append(train_loss_mean)
            loss_rec_all_valid.append(valid_loss_mean)
            print("Epoch %d, Train Loss: %.5f, Valid Loss: %.5f, Valid Stress Loss: %.5f" % (epoch, train_loss_mean, valid_loss_mean, valid_loss_mean_2))
            torch.save(Code_predictor.state_dict(), "./params/" + type + "_Continuum_last.pth")


if __name__ == '__main__':
    train_continuum(type="BM")