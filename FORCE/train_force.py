import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import numpy as np
from FORCE.network import Predictor
from torch.utils.data import Dataset, DataLoader
import os


class PreDataloader(Dataset):
    def __init__(self, file_dir, type, vect_length):
        self.file_dir = file_dir
        self.all_files = os.listdir(file_dir)
        self.all_names_BM = [os.path.join(file_dir, file) for file in self.all_files if (file.endswith("STRESS.npy") and file.startswith(type))]
        self.all_names = [file for file in self.all_files if (file.endswith("STRESS.npy") and file.startswith(type))]
        self.vect_length = vect_length


    def __len__(self):
        return len(self.all_names)

    def __getitem__(self, item):
        BM_fname = self.all_names_BM[item]
        name = self.all_names[item]
        BM_Stress_vect = torch.from_numpy(np.load(BM_fname)[:, 0:self.vect_length]).cuda()

        # print(name)
        name_list = name.split("_")
        temperature = int(name_list[1])
        pressure = int(name_list[2])
        displacement = float(name_list[3])

        if pressure==0:
            temp_FD_fname = r"E:\Data\DATASET\SealDigitTwin\STRESS\LOAD\NPY\\FD_"+ str(temperature) + "_0_16.2"+"_FD.npy"
            array = np.load(temp_FD_fname).T
            pos = np.argmin(np.abs(array[:, 0] - displacement))
            load = array[pos, 2]
            load_tensor = torch.tensor(load, dtype=torch.float32).cuda()
            
        else:
            temp_FD_fname = r"E:\Data\DATASET\SealDigitTwin\STRESS\LOAD\NPY\\FD_" + str(temperature) + "_" + str(pressure) + "_" + str(displacement) + "_FD.npy"
            array = np.load(temp_FD_fname).T
            load = array[0, 2]
            load_tensor = torch.tensor(load, dtype=torch.float32).cuda()

        # BM_vect = np.load(BM_fname)
        # NB_vect = np.load(NB_fname)
        # PR_vect = np.load(PR_fname)
        # BM_vect_torch = torch.from_numpy(BM_vect).cuda()
        # BM_vect_torch_m = torch.transpose(torch.transpose(BM_vect_torch, 0, 1) * self.ratio_vect, 0, 1)
        # NB_vect_torch = torch.from_numpy(NB_vect).cuda()
        # NB_vect_torch_m = torch.transpose(torch.transpose(NB_vect_torch, 0, 1) * self.ratio_vect, 0, 1)
        # PR_vect_torch = torch.from_numpy(PR_vect).cuda()
        # PR_vect_torch_m = torch.transpose(torch.transpose(PR_vect_torch, 0, 1) * self.ratio_vect, 0, 1)
        return BM_Stress_vect, load_tensor, [temperature, pressure, displacement] # torch.from_numpy(np.array([temperature, pressure, displacement], dtype="float32")).cuda(),


def train_predictor(type="BM", save=False, display=False, step=10):
    trainset_dir = r"E:\Data\DATASET\SealDigitTwin\STRESS\TRAIN\\"
    validset_dir = r"E:\Data\DATASET\SealDigitTwin\STRESS\VALID\\"

    lr = 1e-5
    if type == "BM":
        vect_length = 15184
    elif type == "NB":
        vect_length = 13120
    else:
        vect_length = 26016

    # vect_length = 100
    
    train_dl = DataLoader(PreDataloader(trainset_dir, type, vect_length=vect_length),
                          batch_size=1,
                          shuffle=True)
    valid_dl = DataLoader(PreDataloader(validset_dir, type, vect_length=vect_length),
                          batch_size=1,
                          shuffle=True)

    predictor = Predictor(in_channel=6, vect_length=vect_length, num_layer=2).cuda()

    if os.path.exists("./params1019/" + type + "_predictor_last.pth"):
        checkpoint = torch.load("./params1019/" + type + "_predictor_last.pth")
        predictor.load_state_dict(checkpoint)
        results_rec = np.loadtxt("./params1019/" + type + "_training_rec.txt")
        loss_rec_all_train = results_rec[:, 1].tolist()
        loss_rec_all_valid = results_rec[:, 2].tolist()
        results_rec = results_rec.tolist()
    else:
        loss_rec_all_valid = [10000]
        loss_rec_all_train = [10000]
        results_rec = []

    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
    epoch_n = 1000
    loss_fn = torch.nn.MSELoss(reduction="sum")

    loss_rec = []
    for i in range(epoch_n-len(results_rec)*step):
        if i == 0:
            epoch = len(results_rec)*step
        epoch += 1
        predictor.train()

        if epoch % 200 == 199:
            lr = lr / 2
            optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)

        for i, (vects, load, tpd) in enumerate(train_dl):
            predictor.zero_grad()
            predicted = predictor(vects.clone())
            loss = loss_fn(predicted, load)
            loss.backward()
            optimizer.step()
            loss_rec.append(loss.item())

        if epoch % step == 0 or epoch == epoch_n - 1:
            train_loss_mean = np.mean(np.array(loss_rec))
            loss_rec_all_train.append(train_loss_mean)
            loss_rec = []
            for i, (vects, load, tpd) in enumerate(valid_dl):
                predicted = predictor(vects.clone())
                # if i%10 == 0:
                #     print(load[0].item(), predicted[0].item())
                loss = loss_fn(predicted, load)
                loss_rec.append(loss.item())

            valid_loss_mean = np.mean(np.array(loss_rec))
            results_rec.append([lr, train_loss_mean, valid_loss_mean])

            if valid_loss_mean < min(loss_rec_all_valid):
                torch.save(predictor.state_dict(), "./params1019/" + type + "_predictor_best.pth")
            np.savetxt("./params1019/" + type + "_training_rec.txt", np.array(results_rec))
            loss_rec_all_valid.append(valid_loss_mean)
            print("Epoch %d, Train Loss: %.5f, Valid Loss: %.5f"%(epoch, train_loss_mean, valid_loss_mean))
            torch.save(predictor.state_dict(), "./params1019/" + type + "_predictor_last.pth")


if __name__ == '__main__':
    train_predictor(type="BM")
    # train_predictor(type="NB")
    # train_predictor(type="PR")