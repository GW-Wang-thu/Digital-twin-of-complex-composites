import torch
import numpy as np
from COORD.network import AUTOCODER_array, Predictor, VAE
from torch.utils.data import Dataset, DataLoader
import os
import gpytorch
import time


class PreDataloader(Dataset):
    def __init__(self, file_dir, type="BM", array_mode=False):
        try:
            self.all_files = os.listdir(file_dir)
            self.all_names = [file for file in self.all_files if (file.endswith("Coord.npy") and file.startswith(type))]
            self.all_inputs = [os.path.join(file_dir, file) for file in self.all_files if (file.endswith("Coord.npy") and file.startswith(type))]
        except:
            self.all_inputs = file_dir
            self.all_names = [i.split('\\')[-1] for i in file_dir]
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


class Coder():
    def __init__(self, type):
        if type == "BM":
            vect_length = 15184
        elif type == "NB":
            vect_length = 13120
        else:
            vect_length = 26016
        self.autocoder = AUTOCODER_array(vect_length=vect_length, code_length=24, num_layer=3).cuda()
        self.autocoder.load_state_dict(torch.load("./models/" + type + "_coder_best.pth"))
        self.autocoder.eval()

    def decode_array(self, array):
        array = torch.from_numpy(array.astype("float32")).cuda()
        x_rec = []
        y_rec = []
        z_rec = []

        for i in range(array.shape[0]):
            temp_code = array[i, :].unsqueeze(0)
            decoded_coord = self.autocoder(temp_code, coder=1)[0, :, :].detach().cpu().numpy()
            x_coord = decoded_coord[0, :]
            y_coord = decoded_coord[1, :]
            z_coord = decoded_coord[2, :]

            x_rec.append(x_coord)
            y_rec.append(y_coord)
            z_rec.append(z_coord)

        return x_rec, y_rec, z_rec

    def encode_any(self, vect_torch):
        code = self.autocoder(vect_torch, coder=0)[0, :].detach().cpu().numpy()
        return code

    def decode_any(self, vect):
        vect = torch.from_numpy(vect).cuda()
        temp_code = vect.unsqueeze(0)
        decoded_coord = self.autocoder(temp_code, coder=1)[0, :].detach().cpu().numpy()
        coord = decoded_coord
        return coord[0, :], coord[1, :], coord[2, :]

    def encode_decode_any(self, stress_array):
        vect_torch = torch.from_numpy(stress_array).cuda()
        temp_code = torch.transpose(torch.reshape(vect_torch, shape=(vect_torch.shape[0] // 3, 3)), dim0=0, dim1=1).unsqueeze(0)
        code = self.autocoder(temp_code, coder=0)
        decoded_stress = self.autocoder(code, coder=1)[0, :].detach().cpu().numpy()
        stress = decoded_stress
        return stress


class Coder_VAE():
    def __init__(self, type):
        if type == "BM":
            vect_length = 15184
        elif type == "NB":
            vect_length = 13120
        else:
            vect_length = 26016
        self.autocoder = VAE(vect_length=vect_length, code_length=24, num_layer=2).cuda()
        self.autocoder.load_state_dict(torch.load("../../COORD/params1/" + type + "_coder_best.pth"))
        self.autocoder.eval()

    def decode_array(self, array):
        array = torch.from_numpy(array.astype("float32")).cuda()
        x_rec = []
        y_rec = []
        z_rec = []

        for i in range(array.shape[0]):
            temp_code = array[i, :].unsqueeze(0)
            decoded_coord = self.autocoder.decode(temp_code)[0, :, :].detach().cpu().numpy()
            x_coord = decoded_coord[0, :]
            y_coord = decoded_coord[1, :]
            z_coord = decoded_coord[2, :]

            x_rec.append(x_coord)
            y_rec.append(y_coord)
            z_rec.append(z_coord)

        return x_rec, y_rec, z_rec

    def encode_any(self, vect_torch):
        code, _ = self.autocoder.encode(vect_torch)
        return code[0, :].detach().cpu().numpy()

    def decode_any(self, vect):
        vect = torch.from_numpy(vect).cuda()
        temp_code = vect.unsqueeze(0)
        decoded_coord = self.autocoder.decode(temp_code)[0, :].detach().cpu().numpy()
        coord = decoded_coord
        return coord[0, :], coord[1, :], coord[2, :]

    def encode_decode_any(self, stress_array):
        vect_torch = torch.from_numpy(stress_array).cuda()
        temp_code = torch.transpose(torch.reshape(vect_torch, shape=(vect_torch.shape[0] // 3, 3)), dim0=0, dim1=1).unsqueeze(0)
        decoded_stress = self.autocoder(temp_code)[0, :].detach().cpu().numpy()
        stress = decoded_stress
        return stress


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_task):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), #LinearMean(input_size=3)
            num_tasks=num_task,
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(),
            num_tasks=num_task,
            rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class Sampler():
    def __init__(self, num_sample):
        self.num_sample = num_sample

    def sample_gaussian(self, mean_vect, sigma_vect):
        return np.random.normal(loc=mean_vect, scale=sigma_vect)

    def sample(self, mean_vect, sigma_vect):
        rec = []
        for i in range(self.num_sample):
            temp_sample = self.sample_gaussian(mean_vect, sigma_vect)
            rec.append(temp_sample)
        all_coord_codes = np.array(rec)
        return all_coord_codes


def train_GP(type, stage, train_code_predict, train_code_encoder, eval_code_predict):
    num_task = 24
    training_iter = 3000

    train_x = torch.from_numpy(train_code_predict)
    train_y = torch.from_numpy(train_code_encoder)
    eval_x = torch.from_numpy(eval_code_predict)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_task)
    model = MultitaskGPModel(train_x, train_y, likelihood, num_task)

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    now = time.perf_counter()
    print("\nTraining GP Model of " + type + " and of stage "+stage)
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        if i % 100 == 0:
            print('\rIter %d/%d - Loss: %.3f' % (
                i + 1, training_iter, loss.item(),
            ), end="\b")
        optimizer.step()
    print("\ntraining time", time.perf_counter() - now)

    model.eval()
    likelihood.eval()
    input = eval_x
    predicted = model(input)
    mean = predicted.mean.detach().cpu().numpy()
    low, up = predicted.confidence_region()
    # eval_shape = (100, 100, 2)
    results = {
        "mean": mean,
        "sigma": (mean-low.detach().cpu().numpy()) / 2,
    }
    return results


def coord_predict_guassian(init_npys_dir, add_npys_list, to_predict_list, name_char="", savedir="./results/", train_init=False, layer="BM"):
    '''Objects'''
    # Coder
    my_Coder = Coder_VAE(type=layer)
    # Predictor
    my_Predictor = Predictor(code_length=24, params_length=3, hiden_length=64).cuda()
    my_Predictor.load_state_dict(torch.load("../../COORD/params1/"+layer+"_Predictor_best.pth"))
    my_Predictor.eval()
    # Dataloader
    my_init_Dataloader = DataLoader(PreDataloader(init_npys_dir, type=layer, array_mode=True), batch_size=1, shuffle=False)
    my_add_Dataloader = DataLoader(PreDataloader(add_npys_list, type=layer, array_mode=True), batch_size=1, shuffle=False)
    my_pre_Dataloader = DataLoader(PreDataloader(to_predict_list, type=layer, array_mode=True), batch_size=1, shuffle=False)
    # Sampler
    my_Sampler = Sampler(num_sample=100)

    '''From NPY to Codes'''
    # BM
    inputs_all_rec = []
    Codes_all_Coder_Rec = []
    Codes_all_Predict_Rec = []
    inputs_init_rec = []
    Codes_init_Coder_Rec = []
    Codes_init_Predict_Rec = []
    inputs_topre_rec = []
    Codes_topre_Coder_Rec = []
    Codes_topre_Predict_Rec = []


    # BM
    for i, [inputs, vects, _] in enumerate(my_init_Dataloader):
        temp_inputs = inputs.detach().cpu().numpy()
        temp_codes_encode = my_Coder.encode_any(vects)
        temp_codes_predicted = my_Predictor(inputs)[0].detach().cpu().numpy()
        inputs_init_rec.append(temp_inputs)
        Codes_init_Coder_Rec.append(temp_codes_encode)
        Codes_init_Predict_Rec.append(temp_codes_predicted)
        inputs_all_rec.append(temp_inputs)
        Codes_all_Coder_Rec.append(temp_codes_encode)
        Codes_all_Predict_Rec.append(temp_codes_predicted)

    # 待预测列表
    for i, [inputs, vects, _] in enumerate(my_pre_Dataloader):
        temp_inputs = inputs.detach().cpu().numpy()
        temp_codes_encode = my_Coder.encode_any(vects)
        temp_codes_predicted = my_Predictor(inputs)[0].detach().cpu().numpy()
        inputs_topre_rec.append(temp_inputs)
        Codes_topre_Coder_Rec.append(temp_codes_encode)
        Codes_topre_Predict_Rec.append(temp_codes_predicted)

    for i, [inputs, vects, _] in enumerate(my_add_Dataloader):

        temp_inputs = inputs.detach().cpu().numpy()
        print("第%d次更新GP模型，添加位移-力测量数据："%(i), temp_inputs)
        temp_codes_encode = my_Coder.encode_any(vects)
        temp_codes_predicted = my_Predictor(inputs)[0].detach().cpu().numpy()
        inputs_all_rec.append(temp_inputs)
        Codes_all_Coder_Rec.append(temp_codes_encode)
        Codes_all_Predict_Rec.append(temp_codes_predicted)

        code_results_aftertrain = train_GP(type="BM", stage="AfterTrain",
                                           train_code_predict=np.array(Codes_all_Predict_Rec),
                                           train_code_encoder=np.array(Codes_all_Coder_Rec),
                                           eval_code_predict=np.array(Codes_topre_Predict_Rec))
        for j in range(len(Codes_topre_Predict_Rec)):
            if not os.path.exists(savedir + str(j)):
                os.mkdir(savedir + str(j))
            temp_aftertrain_code_array_sampled = my_Sampler.sample(mean_vect=code_results_aftertrain["mean"][j, :],
                                                                   sigma_vect=code_results_aftertrain["sigma"][j, :])
            temp_aftertrain_x, temp_aftertrain_y, temp_aftertrain_z = my_Coder.decode_array(temp_aftertrain_code_array_sampled)
            np.save(savedir + str(j) + "\\" + str(i) + "_" + name_char + layer + "_Bayesian_Coord_Predicted_X.npy", np.array(temp_aftertrain_x, dtype='float32'))
            np.save(savedir + str(j) + "\\" + str(i) + "_" + name_char + layer + "_Bayesian_Coord_Predicted_Y.npy", np.array(temp_aftertrain_y, dtype='float32'))
            np.save(savedir + str(j) + "\\" + str(i) + "_" + name_char + layer + "_Bayesian_Coord_Predicted_Z.npy", np.array(temp_aftertrain_z, dtype='float32'))


if __name__ == '__main__':

    pressure_list = [100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100, 95,100,100,
                     100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,
                      95,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]
    a1 = [2.21, 3.02, 4.64, 6.26, 7.07, 8.69, 10.31, 11.12, 12.74, 13.55, 14.36, 15.17] #12
    displacement_list = a1 + \
        [16.0, 17.0, 18.0, 19.0,20.0, 20.0, 20.0, 19.0, # 8
         18.0, 17.0, 16.0,15.17,16.0, 17.17,18.18,19.19, 20.2, 21.0,    # 10
         19.95,18.9,17.85, 16.8,15.17,16.48,17.51,18.54, 19.57,20.6,    # 10
         20.0, 20.0, 19.0, 18.0, 17.0, 16.0] + [a1[-i-1] for i in range(len(a1))]   # 6+12

    pressure_list = [100 - a for a in pressure_list]
    update_list = []
    for i in range(len(pressure_list)):
        update_list.append('E:\Data\DATASET\SealDigitTwin\FINAL\COORD\\add_coords\BM_25_'+str(pressure_list[i])+'_'+str(displacement_list[i])+'_Coord.npy')
    # update_list = [
    #     r'E:\Data\DATASET\SealDigitTwin\FINAL\COORD\add_coords\BM_25_0_16.0_Coord.npy',
    #     r'E:\Data\DATASET\SealDigitTwin\FINAL\COORD\add_coords\BM_25_0_17.0_Coord.npy',
    #     r'E:\Data\DATASET\SealDigitTwin\FINAL\COORD\add_coords\BM_25_0_18.0_Coord.npy',
    #     r'E:\Data\DATASET\SealDigitTwin\FINAL\COORD\add_coords\BM_25_0_19.0_Coord.npy',
    #     r'E:\Data\DATASET\SealDigitTwin\FINAL\COORD\add_coords\BM_25_0_20.0_Coord.npy',
    #     r'E:\Data\DATASET\SealDigitTwin\FINAL\COORD\add_coords\BM_25_5_20.0_Coord.npy',
    #     r'E:\Data\DATASET\SealDigitTwin\FINAL\COORD\add_coords\BM_25_0_21.0_Coord.npy',
    #     r'E:\Data\DATASET\SealDigitTwin\FINAL\COORD\add_coords\BM_25_5_20.6_Coord.npy',
    #     r'E:\Data\DATASET\SealDigitTwin\FINAL\COORD\add_coords\BM_25_0_20.6_Coord.npy',
    #     r'E:\Data\DATASET\SealDigitTwin\FINAL\COORD\add_coords\BM_25_0_19.95_Coord.npy',
    #     r'E:\Data\DATASET\SealDigitTwin\FINAL\COORD\add_coords\BM_25_0_18.54_Coord.npy',
    #     r'E:\Data\DATASET\SealDigitTwin\FINAL\COORD\add_coords\BM_25_0_17.51_Coord.npy',
    #     r'E:\Data\DATASET\SealDigitTwin\FINAL\COORD\add_coords\BM_25_0_16.8_Coord.npy',
    #     r'E:\Data\DATASET\SealDigitTwin\FINAL\COORD\add_coords\BM_25_0_16.48_Coord.npy',
    #     r'E:\Data\DATASET\SealDigitTwin\FINAL\COORD\add_coords\BM_25_0_17.85_Coord.npy',
    #     r'E:\Data\DATASET\SealDigitTwin\FINAL\COORD\add_coords\BM_25_0_18.9_Coord.npy',
    #     r'E:\Data\DATASET\SealDigitTwin\FINAL\COORD\add_coords\BM_25_0_19.57_Coord.npy',
    #     r'E:\Data\DATASET\SealDigitTwin\FINAL\COORD\add_coords\BM_25_0_20.2_Coord.npy',
    #     r'E:\Data\DATASET\SealDigitTwin\FINAL\COORD\add_coords\BM_25_5_20.2_Coord.npy',
    #     r'E:\Data\DATASET\SealDigitTwin\FINAL\COORD\add_coords\BM_25_0_19.19_Coord.npy',
    #     r'E:\Data\DATASET\SealDigitTwin\FINAL\COORD\add_coords\BM_25_0_18.18_Coord.npy',
    #     r'E:\Data\DATASET\SealDigitTwin\FINAL\COORD\add_coords\BM_25_0_17.17_Coord.npy'
    # ]

    topre_file_list = update_list.copy()

    coord_predict_guassian(init_npys_dir=r"E:\Data\DATASET\SealDigitTwin\COORD\TRAIN\\",
                           add_npys_list=update_list,
                           to_predict_list=topre_file_list,
                           name_char="20221021_",
                           savedir=r"E:\Data\DATASET\SealDigitTwin\FINAL\COORD\results\L1\\",
                           train_init=False,
                           layer="BM")
