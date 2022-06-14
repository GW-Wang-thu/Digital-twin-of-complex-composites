import torch
import numpy as np
from COORD_NL.network import success_coder, Predictor, AUTOCODER_array
from torch.utils.data import Dataset, DataLoader
import os


class COORD_dataloader(Dataset):
    def __init__(self, file_dir, array_mode=False):
        self.all_files = os.listdir(file_dir)
        self.bm_inputs = [os.path.join(file_dir, file) for file in self.all_files if (file.endswith("Coord.npy") and file.startswith("BM"))]
        self.nb_inputs = [os.path.join(file_dir, file) for file in self.all_files if (file.endswith("Coord.npy") and file.startswith("NB"))]
        self.pr_inputs = [os.path.join(file_dir, file) for file in self.all_files if (file.endswith("Coord.npy") and file.startswith("PR"))]
        self.array_mode = array_mode

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
        if self.array_mode:
            return  torch.transpose(torch.reshape(bm_vect_torch, shape=(bm_vect_torch.shape[0]//3, 3)), dim0=0, dim1=1), \
                    torch.transpose(torch.reshape(pr_vect_torch, shape=(pr_vect_torch.shape[0] // 3, 3)), dim0=0, dim1=1), \
                    torch.transpose(torch.reshape(nb_vect_torch, shape=(nb_vect_torch.shape[0] // 3, 3)), dim0=0, dim1=1), \
                    bm_filename


class PreDataloader(Dataset):
    def __init__(self, file_dir, array_mode=True):
        self.all_files = os.listdir(file_dir)
        self.all_bm_names = [file for file in self.all_files if (file.endswith("Coord.npy") and file.startswith("BM"))]
        self.all_bm_inputs = [os.path.join(file_dir, file) for file in self.all_files if (file.endswith("Coord.npy") and file.startswith("BM"))]
        self.all_pr_names = [file for file in self.all_files if (file.endswith("Coord.npy") and file.startswith("PR"))]
        self.all_pr_inputs = [os.path.join(file_dir, file) for file in self.all_files if (file.endswith("Coord.npy") and file.startswith("PR"))]
        self.all_nb_names = [file for file in self.all_files if (file.endswith("Coord.npy") and file.startswith("NB"))]
        self.all_nb_inputs = [os.path.join(file_dir, file) for file in self.all_files if (file.endswith("Coord.npy") and file.startswith("NB"))]
        self.array_mode = array_mode

    def __len__(self):
        return len(self.all_bm_inputs)

    def __getitem__(self, item):
        bm_filename = self.all_bm_inputs[item]
        nb_filename = self.all_nb_inputs[item]
        pr_filename = self.all_pr_inputs[item]
        bm_vect = np.load(bm_filename)
        nb_vect = np.load(nb_filename)
        pr_vect = np.load(pr_filename)

        bm_vect_torch = torch.from_numpy(bm_vect).cuda()
        nb_vect_torch = torch.from_numpy(nb_vect).cuda()
        pr_vect_torch = torch.from_numpy(pr_vect).cuda()
        name = self.all_bm_names[item]
        # print(name)
        name_list = name.split("_")
        temperature = int(name_list[1])
        pressure = int(name_list[2])
        displacement = float(name_list[3])

        if self.array_mode:
            torch.transpose(torch.reshape(bm_vect_torch, shape=(bm_vect_torch.shape[0] // 3, 3)), dim0=0, dim1=1)
            torch.transpose(torch.reshape(pr_vect_torch, shape=(pr_vect_torch.shape[0] // 3, 3)), dim0=0, dim1=1)
            torch.transpose(torch.reshape(nb_vect_torch, shape=(nb_vect_torch.shape[0] // 3, 3)), dim0=0, dim1=1)
            return torch.from_numpy(np.array([temperature, pressure, displacement], dtype="float32")).cuda(), \
                   torch.transpose(torch.reshape(bm_vect_torch, shape=(bm_vect_torch.shape[0] // 3, 3)), dim0=0, dim1=1),\
                   torch.transpose(torch.reshape(pr_vect_torch, shape=(pr_vect_torch.shape[0] // 3, 3)), dim0=0, dim1=1),\
                   torch.transpose(torch.reshape(nb_vect_torch, shape=(nb_vect_torch.shape[0] // 3, 3)), dim0=0, dim1=1), \
                   bm_filename


def train_autocoder(save=False, display=False, step=10):
    trainset_dir = "I:\\DigitRubber_Dataset\\NPY\\COORD\\TRAIN\\"
    validset_dir = "I:\\DigitRubber_Dataset\\NPY\\COORD\\VALID\\"
    if save:
        savedir = "I:\\DigitRubber_Dataset\\NPY\\VALID_RESULTS\\COORD\\NONLAYERED\\"
    train_dl = DataLoader(COORD_dataloader(trainset_dir, array_mode=True),
                          batch_size=1,
                          shuffle=True)
    valid_dl = DataLoader(COORD_dataloader(validset_dir, array_mode=True),
                          batch_size=1,
                          shuffle=True)
    lr = 1e-5

    autocoder_bm = AUTOCODER_array(vect_length=15184, code_length=24, num_layer=3).cuda()       # BM Layer block
    autocoder_pr = AUTOCODER_array(vect_length=26016, code_length=24, num_layer=3).cuda()       # PR Layer block
    autocoder_nb = AUTOCODER_array(vect_length=13120, code_length=24, num_layer=3).cuda()       # NB Layer block
    autocoder_su = success_coder(vect_length=24*3, code_length=72).cuda()                       # concat layer block

    ## uncomment if there is pretrained
    # checkpoint = torch.load("BM_coder_best.pth")
    # autocoder_bm.load_state_dict(checkpoint)
    # checkpoint = torch.load("PR_coder_best.pth")
    # autocoder_pr.load_state_dict(checkpoint)
    # checkpoint = torch.load("NB_coder_best.pth")
    # autocoder_nb.load_state_dict(checkpoint)
    # checkpoint = torch.load("SU_coder_best.pth")
    # autocoder_su.load_state_dict(checkpoint)


    # loss_rec_all_valid = np.loadtxt("loss_valid.txt").tolist()
    # loss_rec_all_train = np.loadtxt("loss_train.txt").tolist()
    loss_rec_all_valid = [1]
    loss_rec_all_train = []

    optimizer_bm = torch.optim.Adam(autocoder_bm.parameters(), lr=lr)
    optimizer_pr = torch.optim.Adam(autocoder_pr.parameters(), lr=lr)
    optimizer_nb = torch.optim.Adam(autocoder_nb.parameters(), lr=lr)
    optimizer_su = torch.optim.Adam(autocoder_su.parameters(), lr=lr)
    epoch_n = 2000
    loss_fn = torch.nn.MSELoss(reduce=True, reduction='mean')
    # step = 1

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
            all_code_output = autocoder_su(all_code_raw)                        # summarized code
            decoded_output_bm = autocoder_bm(all_code_output[:, 0:24], coder=1)     # Decoded BM shape
            decoded_output_pr = autocoder_pr(all_code_output[:, 24:48], coder=1)    # Decoded PR shape
            decoded_output_nb = autocoder_nb(all_code_output[:, 48:], coder=1)      # Decoded NB shape

            autocoder_bm.zero_grad()
            autocoder_pr.zero_grad()
            autocoder_nb.zero_grad()
            autocoder_su.zero_grad()

            loss = loss_fn(decoded_output_bm, vects_bm) + loss_fn(decoded_output_nb, vects_nb) + loss_fn(decoded_output_pr, vects_pr)
            loss.backward()

            optimizer_bm.step()
            optimizer_pr.step()
            optimizer_nb.step()
            optimizer_su.step()         # Update weights all together

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
                loss = loss_fn(decoded_output_bm, vects_bm) + loss_fn(decoded_output_nb, vects_nb) + loss_fn(decoded_output_pr, vects_pr)
                loss_rec.append(loss.item())
                '''for visualization'''
                if display:
                    vect_outputs_bm = decoded_output_bm[0, :].detach().cpu().numpy()
                    vect_outputs_pr = decoded_output_pr[0, :].detach().cpu().numpy()
                    vect_outputs_nb = decoded_output_nb[0, :].detach().cpu().numpy()
                    vect_init_bm = vects_bm[0, :].detach().cpu().numpy()
                    vect_init_pr = vects_pr[0, :].detach().cpu().numpy()
                    vect_init_nb = vects_nb[0, :].detach().cpu().numpy()
                    x_coord = [vect_outputs_bm[3 * i] for i in range(vect_outputs_bm.shape[0] // 3)][0].tolist() + [vect_outputs_pr[3 * i] for i in range(vect_outputs_pr.shape[0] // 3)][0].tolist() + [vect_outputs_nb[3 * i] for i in range(vect_outputs_nb.shape[0] // 3)][0].tolist()
                    y_coord = [vect_outputs_bm[3 * i + 1] for i in range(vect_outputs_bm.shape[0] // 3)][0].tolist() + [vect_outputs_pr[3 * i + 1] for i in range(vect_outputs_pr.shape[0] // 3)][0].tolist() + [vect_outputs_nb[3 * i + 1] for i in range(vect_outputs_nb.shape[0] // 3)][0].tolist()
                    z_coord = [vect_outputs_bm[3 * i + 2] for i in range(vect_outputs_bm.shape[0] // 3)][0].tolist() + [vect_outputs_pr[3 * i + 2] for i in range(vect_outputs_pr.shape[0] // 3)][0].tolist() + [vect_outputs_nb[3 * i + 2] for i in range(vect_outputs_nb.shape[0] // 3)][0].tolist()
                    x_coord_l = [vect_init_bm[3 * i] for i in range(vect_init_bm.shape[0] // 3)][0].tolist() + [vect_init_pr[3 * i] for i in range(vect_init_pr.shape[0] // 3)][0].tolist() + [vect_init_nb[3 * i] for i in range(vect_init_nb.shape[0] // 3)][0].tolist()
                    y_coord_l = [vect_init_bm[3 * i + 1] for i in range(vect_init_bm.shape[0] // 3)][0].tolist() + [vect_init_pr[3 * i + 1] for i in range(vect_init_pr.shape[0] // 3)][0].tolist() + [vect_init_nb[3 * i + 1] for i in range(vect_init_nb.shape[0] // 3)][0].tolist()
                    z_coord_l = [vect_init_bm[3 * i + 2] for i in range(vect_init_bm.shape[0] // 3)][0].tolist() + [vect_init_pr[3 * i + 2] for i in range(vect_init_pr.shape[0] // 3)][0].tolist() + [vect_init_nb[3 * i + 2] for i in range(vect_init_nb.shape[0] // 3)][0].tolist()
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
                if save and epoch > 900:
                    vect_bm_o = decoded_output_bm[0, :].detach().cpu().numpy()
                    temp_name = savedir + "CODER_NONLAYERED_BM_" + file_name[0].split("\\")[-1][3:-4]+".csv"
                    np.savetxt(temp_name, vect_bm_o.T, delimiter=",")
                    vect_pr_o = decoded_output_pr[0, :].detach().cpu().numpy()
                    temp_name = savedir + "CODER_NONLAYERED_PR_" + file_name[0].split("\\")[-1][3:-4]+".csv"
                    np.savetxt(temp_name, vect_pr_o.T, delimiter=",")
                    vect_nb_o = decoded_output_nb[0, :].detach().cpu().numpy()
                    temp_name = savedir + "CODER_NONLAYERED_NB_" + file_name[0].split("\\")[-1][3:-4]+".csv"
                    np.savetxt(temp_name, vect_nb_o.T, delimiter=",")

            valid_loss_mean = np.mean(np.array(loss_rec))

            if valid_loss_mean < min(loss_rec_all_valid):
                torch.save(autocoder_bm.state_dict(), "BM_coder_best.pth")
                torch.save(autocoder_nb.state_dict(), "NB_coder_best.pth")
                torch.save(autocoder_pr.state_dict(), "PR_coder_best.pth")
                torch.save(autocoder_su.state_dict(), "SU_coder_best.pth")
                np.savetxt("loss_valid.txt", np.array(loss_rec_all_valid))
                np.savetxt("loss_train.txt", np.array(loss_rec_all_train))
            loss_rec_all_valid.append(valid_loss_mean)
            print("Epoch %d, Train Loss: %.5f, Valid Loss: %.5f"%(epoch, train_loss_mean, valid_loss_mean))
            # torch.save(autocoder.state_dict(), type + "_coder_last.pth")


def train_predictor(save=False, display=False, step=10):
    trainset_dir = "I:\\DigitRubber_Dataset\\NPY\\COORD\\TRAIN\\"
    validset_dir = "I:\\DigitRubber_Dataset\\NPY\\COORD\\VALID\\"
    train_dl = DataLoader(PreDataloader(trainset_dir, array_mode=True),
                          batch_size=1,
                          shuffle=True)
    valid_dl = DataLoader(PreDataloader(validset_dir, array_mode=True),
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
        savedir = "I:\\DigitRubber_Dataset\\NPY\\VALID_RESULTS\\COORD\\NONLAYERED\\"
    Code_predictor = Predictor(code_length=24*3, params_length=3, hiden_length=64).cuda()
    autocoder_bm = AUTOCODER_array(vect_length=15184, code_length=24, num_layer=3).cuda()
    autocoder_pr = AUTOCODER_array(vect_length=26016, code_length=24, num_layer=3).cuda()
    autocoder_nb = AUTOCODER_array(vect_length=13120, code_length=24, num_layer=3).cuda()
    autocoder_su = success_coder(vect_length=24*3, code_length=72).cuda()

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

    # checkpoint = torch.load("NL_Predictor_last.pth")
    # Code_predictor.load_state_dict(checkpoint)
    optimizer = torch.optim.Adam(Code_predictor.parameters(), lr=lr)
    epoch_n = 2000
    loss_fn = torch.nn.MSELoss(reduce=True, reduction='mean')

    loss_rec_all = [0.1]

    loss_rec = []
    for epoch in range(epoch_n):
        epoch += 1
        Code_predictor.train()

        for i, [inputs, vects_bm, vects_pr, vects_nb, _] in enumerate(train_dl):
            code_outputs_bm = autocoder_bm(vects_bm.clone(), coder=0)
            code_outputs_pr = autocoder_pr(vects_pr.clone(), coder=0)
            code_outputs_nb = autocoder_nb(vects_nb.clone(), coder=0)
            all_code_raw = torch.cat([code_outputs_bm, code_outputs_pr, code_outputs_nb], dim=1)
            all_code_output = autocoder_su(all_code_raw, coder=0)                                   # Aimed code

            outputs = Code_predictor(inputs)                                                        # Predicted code
            # predicted_shape = autocoder(outputs, coder=1)
            Code_predictor.zero_grad()
            loss = loss_fn(all_code_output, outputs)
            loss.backward()
            optimizer.step()

            loss_rec.append(loss.item())

        if epoch % step == 0 or epoch == epoch_n - 1:
            train_loss_mean = np.mean(np.array(loss_rec))
            loss_rec = []
            for i, [inputs, vects_bm, vects_pr, vects_nb, filename] in enumerate(valid_dl):
                code_outputs_bm = autocoder_bm(vects_bm.clone(), coder=0)
                code_outputs_pr = autocoder_pr(vects_pr.clone(), coder=0)
                code_outputs_nb = autocoder_nb(vects_nb.clone(), coder=0)
                all_code_raw = torch.cat([code_outputs_bm, code_outputs_pr, code_outputs_nb], dim=1)
                all_code_output = autocoder_su(all_code_raw, coder=0)

                outputs = Code_predictor(inputs)
                # print(inputs)
                loss = loss_fn(all_code_output, outputs)
                loss_rec.append(loss.item())

                '''For visualization'''
                if display:
                    all_code_output = autocoder_su(outputs, coder=1)
                    decoded_output_bm = autocoder_bm(all_code_output[:, 0:24], coder=1)
                    decoded_output_pr = autocoder_pr(all_code_output[:, 24:48], coder=1)
                    decoded_output_nb = autocoder_nb(all_code_output[:, 48:], coder=1)
                    vect_outputs_bm = decoded_output_bm[0, :].detach().cpu().numpy()
                    vect_outputs_pr = decoded_output_pr[0, :].detach().cpu().numpy()
                    vect_outputs_nb = decoded_output_nb[0, :].detach().cpu().numpy()
                    vect_init_bm = vects_bm[0, :].detach().cpu().numpy()
                    vect_init_pr = vects_pr[0, :].detach().cpu().numpy()
                    vect_init_nb = vects_nb[0, :].detach().cpu().numpy()
                    x_coord = [vect_outputs_bm[3 * i] for i in range(vect_outputs_bm.shape[0] // 3)][0].tolist() + [vect_outputs_pr[3 * i] for i in range(vect_outputs_pr.shape[0] // 3)][0].tolist() + [vect_outputs_nb[3 * i] for i in range(vect_outputs_nb.shape[0] // 3)][0].tolist()
                    y_coord = [vect_outputs_bm[3 * i + 1] for i in range(vect_outputs_bm.shape[0] // 3)][0].tolist() + [vect_outputs_pr[3 * i + 1] for i in range(vect_outputs_pr.shape[0] // 3)][0].tolist() + [vect_outputs_nb[3 * i + 1] for i in range(vect_outputs_nb.shape[0] // 3)][0].tolist()
                    z_coord = [vect_outputs_bm[3 * i + 2] for i in range(vect_outputs_bm.shape[0] // 3)][0].tolist() + [vect_outputs_pr[3 * i + 2] for i in range(vect_outputs_pr.shape[0] // 3)][0].tolist() + [vect_outputs_nb[3 * i + 2] for i in range(vect_outputs_nb.shape[0] // 3)][0].tolist()
                    x_coord_l = [vect_init_bm[3 * i] for i in range(vect_init_bm.shape[0] // 3)][0].tolist() + [vect_init_pr[3 * i] for i in range(vect_init_pr.shape[0] // 3)][0].tolist() + [vect_init_nb[3 * i] for i in range(vect_init_nb.shape[0] // 3)][0].tolist()
                    y_coord_l = [vect_init_bm[3 * i + 1] for i in range(vect_init_bm.shape[0] // 3)][0].tolist() + [vect_init_pr[3 * i + 1] for i in range(vect_init_pr.shape[0] // 3)][0].tolist() + [vect_init_nb[3 * i + 1] for i in range(vect_init_nb.shape[0] // 3)][0].tolist()
                    z_coord_l = [vect_init_bm[3 * i + 2] for i in range(vect_init_bm.shape[0] // 3)][0].tolist() + [vect_init_pr[3 * i + 2] for i in range(vect_init_pr.shape[0] // 3)][0].tolist() + [vect_init_nb[3 * i + 2] for i in range(vect_init_nb.shape[0] // 3)][0].tolist()

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
                if save and epoch > epoch_n-100:
                    all_code_output = autocoder_su(outputs, coder=1)
                    decoded_output_bm = autocoder_bm(all_code_output[:, 0:24], coder=1)
                    decoded_output_pr = autocoder_pr(all_code_output[:, 24:48], coder=1)
                    decoded_output_nb = autocoder_nb(all_code_output[:, 48:], coder=1)
                    vect_bm_o = decoded_output_bm[0, :].detach().cpu().numpy()
                    temp_name = savedir + "PREDICTOR_NONLAYERED_BM_" + filename[0].split("\\")[-1][3:-4] + ".csv"
                    np.savetxt(temp_name, vect_bm_o.T, delimiter=",")
                    vect_pr_o = decoded_output_pr[0, :].detach().cpu().numpy()
                    temp_name = savedir + "PREDICTOR_NONLAYERED_PR_" + filename[0].split("\\")[-1][3:-4] + ".csv"
                    np.savetxt(temp_name, vect_pr_o.T, delimiter=",")
                    vect_nb_o = decoded_output_nb[0, :].detach().cpu().numpy()
                    temp_name = savedir + "PREDICTOR_NONLAYERED_NB_" + filename[0].split("\\")[-1][3:-4] + ".csv"
                    np.savetxt(temp_name, vect_nb_o.T, delimiter=",")

            valid_loss_mean = np.mean(np.array(loss_rec))
            if valid_loss_mean < min(loss_rec_all):
                torch.save(Code_predictor.state_dict(), "NL_Predictor_best.pth")
            loss_rec_all.append(valid_loss_mean)
            print("Epoch %d, Train Loss: %.4f, Valid Loss: %.4f" % (epoch, train_loss_mean, valid_loss_mean))
            torch.save(Code_predictor.state_dict(), "NL_Predictor_last.pth")


if __name__ == '__main__':
    # train_autocoder(save=True, display=False, step=10)
    train_predictor(save=True, display=False, step=10)
    # train_predictor()


