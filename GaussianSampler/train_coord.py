import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import numpy as np
from COORD.network import Predictor, AUTOCODER_array, VAE
from torch.utils.data import Dataset, DataLoader
import os


class COORD_dataloader(Dataset):
    def __init__(self, file_dir, type="BM", array_mode=False):
        self.all_files = os.listdir(file_dir)
        self.all_inputs = [os.path.join(file_dir, file) for file in self.all_files if (file.endswith("Coord.npy") and file.startswith(type))]
        self.type = type
        self.array_mode = array_mode

    def __len__(self):
        return len(self.all_inputs)

    def __getitem__(self, item):
        random_array = np.random.normal(1, 0.02, size=(1, 3)).astype("float32")
        random_array[0, 1] = random_array[0, 1] + (random_array[0, 1] - 1) / 2
        filename = self.all_inputs[item]
        vect = np.load(filename)
        vect_torch = torch.from_numpy(vect).cuda()
        if self.array_mode:
            return torch.transpose(torch.reshape(vect_torch, shape=(vect_torch.shape[0]//3, 3)), dim0=0, dim1=1).cuda(), filename   # * torch.from_numpy(random_array.T)
        else:
            return vect_torch


class PreDataloader(Dataset):
    def __init__(self, file_dir, type="BM", array_mode=False):
        self.all_files = os.listdir(file_dir)
        self.all_names = [file for file in self.all_files if (file.endswith("Coord.npy") and file.startswith(type))]
        self.all_inputs = [os.path.join(file_dir, file) for file in self.all_files if (file.endswith("Coord.npy") and file.startswith(type))]
        self.array_mode = array_mode

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
        if self.array_mode:
            vect_torch = torch.transpose(torch.reshape(vect_torch, shape=(vect_torch.shape[0]//3, 3)), dim0=0, dim1=1)
        return torch.from_numpy(np.array([temperature, pressure, displacement], dtype="float32")).cuda(), vect_torch, filename


def train_autocoder(type="BM", save=False, display=False, step=10):
    trainset_dir = r"E:\Data\DATASET\SealDigitTwin\COORD\TRAIN\\"
    validset_dir = r"E:\Data\DATASET\SealDigitTwin\Results\0_input_npy\COORD_topre\\"
    if save:
        savedir = "I:\\DigitRubber_Dataset\\NPY\\VALID_RESULTS\\COORD\\CODER\\"
    train_dl = DataLoader(COORD_dataloader(trainset_dir, type=type, array_mode=True),
                          batch_size=1,
                          shuffle=True)
    valid_dl = DataLoader(COORD_dataloader(validset_dir, type=type, array_mode=True),
                          batch_size=1,
                          shuffle=True)
    lr = 1e-5
    if type == "BM":
        vect_length = 15184
    elif type == "NB":
        vect_length = 13120
    else:
        vect_length = 26016
    autocoder = VAE(vect_length=vect_length, code_length=24, num_layer=2).cuda()

    if os.path.exists("./params1-noshuffle/" + type + "_coder_last.pth"):
        checkpoint = torch.load("./params1-noshuffle/" + type + "_coder_best.pth")
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
    epoch_n = 500

    loss_rec = []
    for epoch in range(epoch_n):
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
                    x_coord = [vect_outputs[3 * i] for i in range(vect_outputs.shape[0] // 3)]
                    y_coord = [vect_outputs[3 * i + 1] for i in range(vect_outputs.shape[0] // 3)]
                    z_coord = [vect_outputs[3 * i + 2] for i in range(vect_outputs.shape[0] // 3)]
                    x_coord_l = [vect_init[3 * i] for i in range(vect_init.shape[0] // 3)]
                    y_coord_l = [vect_init[3 * i + 1] for i in range(vect_init.shape[0] // 3)]
                    z_coord_l = [vect_init[3 * i + 2] for i in range(vect_init.shape[0] // 3)]
                    import matplotlib.pyplot as plt
                    fig = plt.figure(dpi=128, figsize=(8, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(x_coord, y_coord, z_coord, s=1, cmap="jet", label="decoded")
                    ax.scatter(x_coord_l, y_coord_l, z_coord_l, s=1, cmap="jet", label="Label")
                    ax.set_xlabel('X', fontsize=10)
                    ax.set_ylabel('Y', fontsize=10)
                    ax.set_zlabel('Z', fontsize=10)
                    plt.title("Shapes")
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
    trainset_dir = r"E:\Data\DATASET\SealDigitTwin\COORD\TRAIN\\"
    validset_dir = r"E:\Data\DATASET\SealDigitTwin\COORD\VALID\\"
    train_dl = DataLoader(PreDataloader(trainset_dir, type=type, array_mode=True),
                          batch_size=1,
                          shuffle=True)
    valid_dl = DataLoader(PreDataloader(validset_dir, type=type, array_mode=True),
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

    Code_predictor = Predictor(code_length=24, params_length=3, hiden_length=64).cuda()
    autocoder = VAE(vect_length=vect_length, code_length=24, num_layer=2).cuda()

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
    epoch_n = 500
    loss_fn = torch.nn.MSELoss(reduce=True, reduction='sum')

    for epoch in range(epoch_n):
        epoch += 1
        Code_predictor.train()

        if epoch % 100 == 99:
            lr = lr / 2
            optimizer = torch.optim.Adam(Code_predictor.parameters(), lr=lr)
        # train_data = []
        loss_rec = []
        for i, [inputs, vects, _] in enumerate(train_dl):
            codes, _ = autocoder.encode(vects)
            outputs = Code_predictor(inputs)
            # predicted_shape = autocoder(outputs, coder=1)
            # temp_input_code_line = torch.cat([inputs, codes, outputs], dim=1)[0, :].detach().cpu().numpy()
            # train_data.append(temp_input_code_line)
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
            for i, [inputs, vects, filename] in enumerate(valid_dl):
                codes, _ = autocoder.encode(vects)
                outputs = Code_predictor(inputs)
                # temp_input_code_line = torch.cat([inputs, codes, outputs], dim=1)[0, :].detach().cpu().numpy()
                # valid_data.append(temp_input_code_line)
                # print(inputs)
                shape_outputs = autocoder.decode(outputs)
                loss = loss_fn(codes, outputs)
                loss_rec.append(loss.item())

                '''下面的代码是用于可视化的，训练的时候可以注释掉'''
                if display:
                    vect_outputs = shape_outputs[0, :].detach().cpu().numpy()
                    vect_init = vects[0, :].detach().cpu().numpy()
                    x_coord = [vect_outputs[3 * i] for i in range(vect_outputs.shape[0] // 3)]
                    y_coord = [vect_outputs[3 * i + 1] for i in range(vect_outputs.shape[0] // 3)]
                    z_coord = [vect_outputs[3 * i + 2] for i in range(vect_outputs.shape[0] // 3)]
                    x_coord_l = [vect_init[3 * i] for i in range(vect_init.shape[0] // 3)]
                    y_coord_l = [vect_init[3 * i + 1] for i in range(vect_init.shape[0] // 3)]
                    z_coord_l = [vect_init[3 * i + 2] for i in range(vect_init.shape[0] // 3)]
                    import matplotlib.pyplot as plt
                    fig = plt.figure(dpi=128, figsize=(8, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(x_coord, y_coord, z_coord, cmap="jet", s=1, label="Predicted")
                    ax.scatter(x_coord_l, y_coord_l, z_coord_l, cmap="jet", s=1, label="Label")
                    ax.set_xlabel('X', fontsize=10)
                    ax.set_ylabel('Y', fontsize=10)
                    ax.set_zlabel('Z', fontsize=10)
                    inputs_num = inputs[0, :].detach().cpu().numpy()
                    plt.title("Shapes of T: %d deg, P: %d kPa, D: %.2f mm"%(int(inputs_num[0]), int(inputs_num[1]), inputs_num[2]))
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
    # train_autocoder(type="BM", save=False, display=False, step=10)
    # train_autocoder(type="PR", save=False, display=True, step=10)
    # train_autocoder(type="NB", save=False, display=True, step=10)

    train_predictor(type="BM", save=False, display=False, step=10)
    # train_predictor(type="PR", save=False, display=False, step=10)
    # train_predictor(type="NB", save=False, display=False, step=10)

