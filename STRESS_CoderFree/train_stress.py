import torch
import numpy as np
from STRESS_CoderFree.network import blocked_AUTOCODER, Predictor, blocked_AUTOCODER_pr
from torch.utils.data import Dataset, DataLoader
import os


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
        savedir = "I:\\DigitRubber_Dataset\\NPY\\VALID_RESULTS\\STRESS\\CODERFREE\\"
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

    # checkpoint = torch.load(type + "_Autocoder_best.pth")
    # autocoder.load_state_dict(checkpoint)
    #
    # checkpoint = torch.load(type + "_Predictor_best.pth")
    # Code_predictor.load_state_dict(checkpoint)
    optimizer_pre = torch.optim.Adam(Code_predictor.parameters(), lr=lr)        # optimizer for weights in predictor
    optimizer_deco = torch.optim.Adam(autocoder.parameters(), lr=lr)            # optimizer for weights in Decoder

    epoch_n = 2
    loss_fn = torch.nn.MSELoss(reduce=True, reduction='mean')

    loss_rec_all = [10]

    loss_rec = []
    for epoch in range(epoch_n):
        epoch += 1
        Code_predictor.train()

        for i, [inputs, vects, _] in enumerate(train_dl):
            outputs = Code_predictor(inputs).unsqueeze(1)       # Predicted code
            predicted_shape = autocoder(outputs, coder=1)       # Decoder only
            Code_predictor.zero_grad()
            autocoder.zero_grad()
            loss = loss_fn(predicted_shape, vects)
            loss.backward()
            optimizer_pre.step()                                # update weights in Predictor block
            optimizer_deco.step()                               # update weights in Decoder block

            loss_rec.append(loss.item())

        if epoch > 300:
            lr = 1e-6
            optimizer_pre = torch.optim.Adam(Code_predictor.parameters(), lr=lr)
            optimizer_deco = torch.optim.Adam(autocoder.parameters(), lr=lr)

        if epoch % step == 0 or epoch == epoch_n - 1:
            train_loss_mean = np.mean(np.array(loss_rec))
            loss_rec = []
            for i, [inputs, vects, file_name] in enumerate(valid_dl):
                outputs = Code_predictor(inputs).unsqueeze(1)
                stress_outputs = autocoder(outputs, coder=1)
                loss = loss_fn(stress_outputs, vects)
                loss_rec.append(loss.item())

                '''For visualization'''
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
                    temp_name = savedir + "CODERFREE_" + file_name[0].split("\\")[-1][:-4] + ".csv"
                    np.savetxt(temp_name, vect_outputs, delimiter=",")

            valid_loss_mean = np.mean(np.array(loss_rec))
            if valid_loss_mean < min(loss_rec_all):
                torch.save(Code_predictor.state_dict(), type + "_Predictor_best.pth")
                torch.save(autocoder.state_dict(), type + "_Autocoder_best.pth")
            loss_rec_all.append(valid_loss_mean)
            print("Epoch %d, Train Loss: %.4f, Valid Loss: %.4f" % (epoch, train_loss_mean, valid_loss_mean))
            torch.save(Code_predictor.state_dict(), type + "_Predictor_last.pth")
            torch.save(autocoder.state_dict(), type + "_Autocoder_last.pth")


if __name__ == '__main__':
    train_predictor(type="BM", save=True, display=False, step=10)
    # train_predictor(type="NB", save=True, display=False, step=10)
    # train_predictor(type="PR", save=True, display=False, step=10)

