import torch
import numpy as np
from STRESS_NL.network import blocked_AUTOCODER, Predictor, blocked_AUTOCODER_pr, AUTOCODER_su
from torch.utils.data import Dataset, DataLoader
import os


class STRESS_dataloader(Dataset):
    def __init__(self, file_dir, ratio_vect=[4.0, 1.0, 4.0, 10.0, 20.0, 8.0]):
        self.all_files = os.listdir(file_dir)
        self.bm_inputs = [os.path.join(file_dir, file) for file in self.all_files if (file.endswith("STRESS.npy") and file.startswith("BM"))]
        self.pr_inputs = [os.path.join(file_dir, file) for file in self.all_files if (file.endswith("STRESS.npy") and file.startswith("PR"))]
        self.nb_inputs = [os.path.join(file_dir, file) for file in self.all_files if (file.endswith("STRESS.npy") and file.startswith("NB"))]
        self.ratio_vect = torch.from_numpy(np.array(ratio_vect, dtype="float32")).cuda()

    def __len__(self):
        return len(self.bm_inputs)

    def __getitem__(self, item):
        bm_filename = self.bm_inputs[item]
        nb_filename = self.nb_inputs[item]
        pr_filename = self.pr_inputs[item]
        bm_vect = np.load(bm_filename)
        nb_vect = np.load(nb_filename)
        pr_vect = np.load(pr_filename)
        bm_vect_torch = torch.from_numpy(bm_vect).cuda()
        nb_vect_torch = torch.from_numpy(nb_vect).cuda()
        pr_vect_torch = torch.from_numpy(pr_vect).cuda()

        return bm_vect_torch, \
               pr_vect_torch, \
               nb_vect_torch, \
               bm_filename


class PreDataloader(Dataset):
    def __init__(self, file_dir, ratio_vect=[4.0, 1.0, 4.0, 10.0, 20.0, 8.0]):
        self.all_files = os.listdir(file_dir)
        self.all_names = [file for file in self.all_files if (file.endswith("STRESS.npy"))]
        self.all_inputs_BM = [os.path.join(file_dir, file) for file in self.all_names if (file.startswith("BM"))]
        self.all_inputs_NB = [os.path.join(file_dir, file) for file in self.all_names if (file.startswith("NB"))]
        self.all_inputs_PR = [os.path.join(file_dir, file) for file in self.all_names if (file.startswith("PR"))]
        self.ratio_vect = torch.from_numpy(np.array(ratio_vect, dtype="float32")).cuda()

    def __len__(self):
        return len(self.all_inputs_BM)

    def __getitem__(self, item):

        bm_filename = self.all_inputs_BM[item]
        nb_filename = self.all_inputs_NB[item]
        pr_filename = self.all_inputs_PR[item]
        bm_vect = np.load(bm_filename)
        nb_vect = np.load(nb_filename)
        pr_vect = np.load(pr_filename)

        bm_vect_torch = torch.from_numpy(bm_vect).cuda()
        nb_vect_torch = torch.from_numpy(nb_vect).cuda()
        pr_vect_torch = torch.from_numpy(pr_vect).cuda()
        name = self.all_names[item]
        # print(name)
        name_list = name.split("_")
        temperature = int(name_list[1])
        pressure = int(name_list[2])
        displacement = float(name_list[3])
        # vect_torch_m = torch.transpose(torch.transpose(vect_torch, 0, 1) * self.ratio_vect, 0, 1)
        return torch.from_numpy(np.array([temperature, pressure, displacement], dtype="float32")).cuda(), \
               bm_vect_torch, \
               pr_vect_torch, \
               nb_vect_torch, \
               bm_filename


def train_autocoder(save=True, display=False, step=10):
    trainset_dir = "I:\\DigitRubber_Dataset\\NPY\\STRESS\\TRAIN\\"
    validset_dir = "I:\\DigitRubber_Dataset\\NPY\\STRESS\\VALID\\"
    ratio_train = True
    if save:
        savedir = "I:\\DigitRubber_Dataset\\NPY\\VALID_RESULTS\\STRESS\\CODER\\"

    if ratio_train:
        ratio_vect = [4.5, 1.0, 4.5, 12.0, 25.0, 8.0]
    else:
        ratio_vect = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    train_dl = DataLoader(STRESS_dataloader(trainset_dir, ratio_vect=ratio_vect),
                          batch_size=1,
                          shuffle=True)
    valid_dl = DataLoader(STRESS_dataloader(validset_dir, ratio_vect=ratio_vect),
                          batch_size=1,
                          shuffle=True)
    lr = 1e-4
    autocoder_bm = blocked_AUTOCODER(vect_length=15184, code_length=72, num_layer=3, in_channel=6).cuda()
    autocoder_nb = blocked_AUTOCODER(vect_length=13120, code_length=72, num_layer=3, in_channel=6).cuda()
    autocoder_pr = blocked_AUTOCODER_pr(vect_length=26016, code_length=72, num_layer=3, in_channel=6).cuda()
    autocoder_su = AUTOCODER_su(code_length=72 * 3)

    loss_rec_all_valid = [1]
    loss_rec_all_train = []

    optimizer_bm = torch.optim.Adam(autocoder_bm.parameters(), lr=lr)
    optimizer_pr = torch.optim.Adam(autocoder_pr.parameters(), lr=lr)
    optimizer_nb = torch.optim.Adam(autocoder_nb.parameters(), lr=lr)
    optimizer_su = torch.optim.Adam(autocoder_su.parameters(), lr=lr)

    epoch_n = 2000
    loss_fn = torch.nn.MSELoss(reduce=True, reduction='mean')

    loss_rec = []
    for epoch in range(epoch_n):
        epoch += 1
        autocoder_bm.train()
        autocoder_nb.train()
        autocoder_pr.train()
        autocoder_su.train()

        for i, (vects_bm, vects_pr, vects_nb, _) in enumerate(train_dl):
            code_outputs_bm = autocoder_bm(vects_bm.clone(), coder=0)           # BM Code
            code_outputs_pr = autocoder_pr(vects_pr.clone(), coder=0)           # PR Code
            code_outputs_nb = autocoder_nb(vects_nb.clone(), coder=0)           # NB Code
            all_code_raw = torch.cat([code_outputs_bm, code_outputs_pr, code_outputs_nb], dim=1)    # Concat code
            all_code_output = autocoder_su(all_code_raw)  # summarized code
            decoded_output_bm = autocoder_bm(all_code_output[:, 0:24*3], coder=1)       # Decoded BM Stress
            decoded_output_pr = autocoder_pr(all_code_output[:, 24*3:48*3], coder=1)    # Decoded PR Stress
            decoded_output_nb = autocoder_nb(all_code_output[:, 48*3:], coder=1)        # Decoded NB Stress

            autocoder_bm.zero_grad()
            autocoder_pr.zero_grad()
            autocoder_nb.zero_grad()
            autocoder_su.zero_grad()

            loss = loss_fn(decoded_output_bm, vects_bm) + loss_fn(decoded_output_nb, vects_nb) + loss_fn(
                decoded_output_pr, vects_pr)
            loss.backward()

            optimizer_bm.step()
            optimizer_pr.step()
            optimizer_nb.step()
            optimizer_su.step()  # Update weights all together

            loss_rec.append(loss.item())

        if epoch % step == 0 or epoch == epoch_n - 1:
            train_loss_mean = np.mean(np.array(loss_rec))
            loss_rec_all_train.append(train_loss_mean)
            loss_rec = []
            for i, (vects_bm, vects_pr, vects_nb, file_name) in enumerate(valid_dl):
                code_outputs_bm = autocoder_bm(vects_bm.clone(), coder=0)
                code_outputs_pr = autocoder_pr(vects_pr.clone(), coder=0)
                code_outputs_nb = autocoder_nb(vects_nb.clone(), coder=0)
                all_code_raw = torch.cat([code_outputs_bm, code_outputs_pr, code_outputs_nb], dim=1)
                all_code_output = autocoder_su(all_code_raw)
                decoded_output_bm = autocoder_bm(all_code_output[:, 0:24], coder=1)
                decoded_output_pr = autocoder_pr(all_code_output[:, 24:48], coder=1)
                decoded_output_nb = autocoder_nb(all_code_output[:, 48:], coder=1)
                loss = loss_fn(decoded_output_bm, vects_bm) + loss_fn(decoded_output_nb, vects_nb) + loss_fn(
                    decoded_output_pr, vects_pr)
                loss_rec.append(loss.item())

            valid_loss_mean = np.mean(np.array(loss_rec))

            if valid_loss_mean < min(loss_rec_all_valid):
                torch.save(autocoder_bm.state_dict(), "BM_coder_best.pth")
                torch.save(autocoder_nb.state_dict(), "NB_coder_best.pth")
                torch.save(autocoder_pr.state_dict(), "PR_coder_best.pth")
                torch.save(autocoder_su.state_dict(), "SU_coder_best.pth")
                np.savetxt("loss_valid.txt", np.array(loss_rec_all_valid))
                np.savetxt("loss_train.txt", np.array(loss_rec_all_train))
            loss_rec_all_valid.append(valid_loss_mean)
            print("Epoch %d, Train Loss: %.5f, Valid Loss: %.5f" % (epoch, train_loss_mean, valid_loss_mean))


def train_predictor(save=True, display=True, step=10):
    trainset_dir = "I:\\DigitRubber_Dataset\\NPY\\STRESS\\TRAIN\\"
    validset_dir = "I:\\DigitRubber_Dataset\\NPY\\STRESS\\VALID\\"
    ratio_train = True
    type = "NB"

    if save:
        savedir = "I:\\DigitRubber_Dataset\\NPY\\VALID_RESULTS\\STRESS\\NONLAYERED\\"
    if ratio_train:
        ratio_vect = [4.5, 1.0, 4.5, 12.0, 25.0, 8.0]
    else:
        ratio_vect = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    train_dl = DataLoader(PreDataloader(trainset_dir, ratio_vect=ratio_vect),
                          batch_size=1,
                          shuffle=True)
    valid_dl = DataLoader(PreDataloader(validset_dir, ratio_vect=ratio_vect),
                          batch_size=1,
                          shuffle=True)
    lr = 1e-4
    Code_predictor = Predictor(code_length=72*3, params_length=3, hiden_length=72*2).cuda()
    optimizer = torch.optim.Adam(Code_predictor.parameters(), lr=lr)

    autocoder_bm = blocked_AUTOCODER(vect_length=15184, code_length=72, num_layer=3, in_channel=6).cuda()
    autocoder_nb = blocked_AUTOCODER(vect_length=13120, code_length=72, num_layer=3, in_channel=6).cuda()
    autocoder_pr = blocked_AUTOCODER_pr(vect_length=26016, code_length=72, num_layer=3, in_channel=6).cuda()
    autocoder_su = AUTOCODER_su(code_length=72 * 3)

    checkpoint = torch.load("BM_coder_best.pth")
    autocoder_bm.load_state_dict(checkpoint)
    autocoder_bm.eval()
    checkpoint = torch.load("PR_coder_best.pth")
    autocoder_pr.load_state_dict(checkpoint)
    autocoder_pr.eval()
    checkpoint = torch.load("NB_coder_best.pth")
    autocoder_nb.load_state_dict(checkpoint)
    autocoder_nb.eval()
    checkpoint = torch.load("SU_coder_best.pth")
    autocoder_su.load_state_dict(checkpoint)
    autocoder_su.eval()

    epoch_n = 2000
    loss_fn = torch.nn.MSELoss(reduce=True, reduction='mean')

    loss_rec_all = [10]

    loss_rec = []
    for epoch in range(epoch_n):
        epoch += 1
        Code_predictor.train()

        for i, [inputs, vects_bm, vects_pr, vects_nb, _] in enumerate(train_dl):
            code_outputs_bm = autocoder_bm(vects_bm.clone(), coder=0)
            code_outputs_pr = autocoder_pr(vects_pr.clone(), coder=0)
            code_outputs_nb = autocoder_nb(vects_nb.clone(), coder=0)
            all_code_raw = torch.cat([code_outputs_bm, code_outputs_pr, code_outputs_nb], dim=1)
            all_code_output = autocoder_su(all_code_raw, coder=2)  # Encoder mode, Aimed code
            outputs = Code_predictor(inputs)  # Predicted code
            # predicted_shape = autocoder(outputs, coder=1)
            Code_predictor.zero_grad()
            loss = loss_fn(all_code_output, outputs)
            loss.backward()
            optimizer.step()

        if epoch > 300:
            lr = 1e-6
            optimizer_pre = torch.optim.Adam(Code_predictor.parameters(), lr=lr)

        if epoch % step == 0 or epoch == epoch_n - 1:
            train_loss_mean = np.mean(np.array(loss_rec))
            loss_rec = []
            for i, [inputs, vects_bm, vects_pr, vects_nb, _] in enumerate(valid_dl):
                code_outputs_bm = autocoder_bm(vects_bm.clone(), coder=0)
                code_outputs_pr = autocoder_pr(vects_pr.clone(), coder=0)
                code_outputs_nb = autocoder_nb(vects_nb.clone(), coder=0)
                all_code_raw = torch.cat([code_outputs_bm, code_outputs_pr, code_outputs_nb], dim=1)
                all_code_output = autocoder_su(all_code_raw, coder=0)

                outputs = Code_predictor(inputs)
                # print(inputs)
                loss = loss_fn(all_code_output, outputs)
                loss_rec.append(loss.item())

            valid_loss_mean = np.mean(np.array(loss_rec))
            if valid_loss_mean < min(loss_rec_all):
                torch.save(Code_predictor.state_dict(), "NL_Predictor_best.pth")
                pass
                # torch.save(autocoder.state_dict(), "NL_Autocoder_best.pth")
            loss_rec_all.append(valid_loss_mean)
            print("Epoch %d, Train Loss: %.4f, Valid Loss: %.4f" % (epoch, train_loss_mean, valid_loss_mean))
            torch.save(Code_predictor.state_dict(), "NL_Predictor_last.pth")


if __name__ == '__main__':
    train_autocoder(save=True, display=False, step=10)
    train_predictor(save=True, display=True, step=10)
    # train_autocoder(type="NB", save=True, display=False, step=10)
    # train_predictor(type="NB", save=True, display=False, step=10)
    # train_autocoder(type="PR", save=True, display=False, step=10)
    # train_predictor(save=True, display=False, step=2)

