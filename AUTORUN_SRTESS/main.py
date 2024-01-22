import torch
import numpy as np
from AutoRun.STRESS.network import Predictor, VAE, Predictor_Force, Predictor_Force_DP
from torch.utils.data import Dataset, DataLoader
import os
import gpytorch
import time


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


class Coder_VAE():
    def __init__(self, type, ratio_vect):
        if type == "BM":
            vect_length = 15184
        elif type == "NB":
            vect_length = 13120
        else:
            vect_length = 26016
        self.ratio_vect = ratio_vect
        self.ratio_vect_cuda = torch.from_numpy(np.array(ratio_vect, dtype="float32")).cuda()
        self.autocoder = VAE(vect_length=vect_length, code_length=72, num_layer=2, in_channel=6).cuda()
        # self.autocoder.load_state_dict(torch.load("../../STRESS/params1/" + type + "_coder_best.pth"))
        self.autocoder.load_state_dict(torch.load(r"D:\Codes\DigitTwinRubber_3D\AutoRun\STRESS\models\BM_coder_best.pth"))
        self.autocoder.eval()

    def decode_array(self, array):
        array = torch.from_numpy(array.astype("float32")).cuda()
        s11_rec = []
        s22_rec = []
        s33_rec = []
        s12_rec = []
        s13_rec = []
        s23_rec = []

        for i in range(array.shape[0]):
            temp_code = array[i, :].unsqueeze(0).unsqueeze(0)
            decoded_coord = self.autocoder.decode(temp_code)[0,].detach().cpu().numpy()
            s11 = decoded_coord[0, :] / self.ratio_vect[0]
            s22 = decoded_coord[1, :] / self.ratio_vect[1]
            s33 = decoded_coord[2, :] / self.ratio_vect[2]
            s12 = decoded_coord[3, :] / self.ratio_vect[3]
            s13 = decoded_coord[4, :] / self.ratio_vect[4]
            s23 = decoded_coord[5, :] / self.ratio_vect[5]

            s11_rec.append(s11)
            s22_rec.append(s22)
            s33_rec.append(s33)
            s12_rec.append(s12)
            s13_rec.append(s13)
            s23_rec.append(s23)

        return s11_rec, s22_rec, s33_rec, s12_rec, s13_rec, s23_rec

    def encode_any(self, vect_torch, rt=False):
        # 应力变换
        if rt:
            vect_torch = vect_torch[0]
            vect_torch = torch.transpose(torch.transpose(vect_torch, 0, 1) * self.ratio_vect_cuda, 0, 1).unsqueeze(0)
        code, _ = self.autocoder.encode(vect_torch)
        return code[0, :].detach().cpu().numpy()

    def decode_any(self, vect, mode="vect"):
        vect = torch.from_numpy(vect.astype("float32")).cuda()
        temp_code = vect.unsqueeze(0).unsqueeze(0)
        if mode == "vect":
            decoded_stress = self.autocoder.decode(temp_code)[0, :].detach().cpu().numpy()
            return decoded_stress[0, :] / self.ratio_vect[0], decoded_stress[1, :] / self.ratio_vect[1], decoded_stress[2, :] / self.ratio_vect[2], \
                   decoded_stress[3, :] / self.ratio_vect[3], decoded_stress[4, :] / self.ratio_vect[4], decoded_stress[5, :] / self.ratio_vect[5]
        else:
            decoded_stress = torch.transpose(torch.transpose(self.autocoder.decode(temp_code)[0, ], 0, 1) / self.ratio_vect_cuda, 0, 1).unsqueeze(0)
            return decoded_stress

    def encode_decode_any(self, stress_array):
        vect_torch = torch.from_numpy(stress_array).cuda()
        vect_torch = torch.transpose(torch.transpose(vect_torch, 0, 1) * self.ratio_vect_cuda, 0, 1)
        decoded_stress = self.autocoder(vect_torch)[0, :].detach().cpu().numpy()
        stress = decoded_stress
        return stress


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_task=72):
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


class ForceParser():
    def __init__(self, type):
        if type == "BM":
            vect_length = 15184
        elif type == "NB":
            vect_length = 13120
        else:
            vect_length = 26016
        # self.Force_predictor = Predictor_Force(in_channel=6, vect_length=vect_length, num_layer=2).cuda()
        # checkpoint = torch.load("./models/" + type + "_Force_predictor_best.pth")
        self.Force_predictor = Predictor_Force_DP(in_channel=6, vect_length=vect_length, num_layer=2).cuda()
        checkpoint = torch.load("../../Force/params1019/" + type + "_predictor_best.pth")
        self.Force_predictor.load_state_dict(checkpoint)
        self.Force_predictor.eval()

    def predict_force(self, stress_vect):
        vect_torch = stress_vect
        force_predicted = self.Force_predictor(vect_torch)[0].item()
        return force_predicted


class Sampler():
    def __init__(self, num_sample):
        self.num_sample = num_sample

    def sample_gaussian(self, mean_vect, sigma_vect):
        return np.random.normal(loc=mean_vect, scale=sigma_vect)

    def sample(self, mean_vect, sigma_vect, parser=None, decoder=None, force_label=None):
        rec = []
        if parser is None:
            for i in range(self.num_sample):
                temp_sample = self.sample_gaussian(mean_vect, sigma_vect).astype("float32")
                rec.append(temp_sample)
        else:
            i = 0
            while(len(rec) < self.num_sample and (i < 2000)):
                i += 1
                temp_sample = self.sample_gaussian(mean_vect, sigma_vect).astype("float32")
                force_predicted = parser.predict_force(decoder.decode_any(temp_sample, mode="array"))
                if abs((force_predicted - force_label) / force_label) < 0.08:
                    print(i, "\tavailable, label force: %.2f, predicted force: %.2f"%(force_label, force_predicted))
                    rec.append(temp_sample)
                else:
                    print("unavailable, label force: %.2f, predicted force: %.2f"%(force_label, force_predicted))
            if i >= 1000:
                for i in range(self.num_sample):
                    temp_sample = self.sample_gaussian(mean_vect, sigma_vect).astype("float32")
                    rec.append(temp_sample)
        all_coord_codes = np.array(rec)
        return all_coord_codes


def train_GP(type, stage, train_code_predict, train_code_encoder, eval_code_predict):
    num_task = 72
    training_iter = 500

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


def stress_predict_guassian(init_npys_dir, predict_npys_dir, force_list, save_dir, name_char="", layer="BM"):
    '''Objects'''
    # Coder
    ratio_vect = [4.5, 1.0, 4.5, 12.0, 25.0, 8.0]
    my_Coder = Coder_VAE(type=layer, ratio_vect=ratio_vect)
    # Predictor
    my_Predictor = Predictor(code_length=72, params_length=3, hiden_length=72).cuda()
    # my_Predictor.load_state_dict(torch.load("../../STRESS/params1/"+layer+"_predictor_best.pth"))
    my_Predictor.load_state_dict(torch.load(r"D:\Codes\DigitTwinRubber_3D\AutoRun\STRESS\models\BM_predictor_best.pth"))

    my_Predictor.eval()
    # Dataloader
    my_init_Dataloader = DataLoader(PreDataloader(init_npys_dir, type=layer, ratio_vect=ratio_vect), batch_size=1, shuffle=False)
    my_pre_Dataloader = DataLoader(PreDataloader(predict_npys_dir, type=layer, ratio_vect=ratio_vect), batch_size=1, shuffle=False)
    # Sampler
    my_Sampler = Sampler(num_sample=100)
    # Force Parser
    my_Force_Parser = ForceParser(type=layer)

    '''From NPY to Codes'''
    inputs_init_rec = []
    Codes_init_Coder_Rec = []
    Codes_init_Predict_Rec = []
    inputs_topre_rec = []
    Codes_topre_Coder_Rec = []
    Codes_topre_Predict_Rec = []

    # BM
    for i, [inputs, vects, _] in enumerate(my_init_Dataloader):
        temp_inputs = inputs.detach().cpu().numpy()
        temp_codes_encode = my_Coder.encode_any(vects)[0]
        temp_codes_predicted = my_Predictor(inputs)[0].detach().cpu().numpy()
        inputs_init_rec.append(temp_inputs)
        Codes_init_Coder_Rec.append(temp_codes_encode)
        Codes_init_Predict_Rec.append(temp_codes_predicted)

    for i, [inputs, vects, _] in enumerate(my_pre_Dataloader):
        temp_inputs = inputs.detach().cpu().numpy()
        temp_codes_encode = my_Coder.encode_any(vects)[0]
        temp_codes_predicted = my_Predictor(inputs)[0].detach().cpu().numpy()
        inputs_topre_rec.append(temp_inputs)
        Codes_topre_Coder_Rec.append(temp_codes_encode)
        Codes_topre_Predict_Rec.append(temp_codes_predicted)

    '''Train and predict using INIT data'''
    # BM
    code_results_pretrain = train_GP(type="BM", stage="PreTrain",
                                        train_code_predict=np.array(Codes_init_Predict_Rec),
                                        train_code_encoder=np.array(Codes_init_Coder_Rec),
                                        eval_code_predict=np.array(Codes_topre_Predict_Rec))

    '''Sample and Decode to COORD'''

    # PRETRAIN
    # Bayesian
    for i in range(code_results_pretrain["mean"].shape[0]):
        temp_pretrain_code_array_sampled = my_Sampler.sample(mean_vect=code_results_pretrain["mean"][i, :],
                                                                sigma_vect=code_results_pretrain["sigma"][i, :],
                                                                parser=my_Force_Parser,
                                                                decoder=my_Coder,
                                                                force_label=force_list[i])

        temp_s11, temp_s22, temp_s33, temp_s12, temp_s13, temp_s23 = my_Coder.decode_array(temp_pretrain_code_array_sampled)
        np.savetxt(save_dir+str(i)+"_"+name_char+layer+"_Bayesian_s11.csv", np.array(temp_s11), delimiter=",")
        np.savetxt(save_dir+str(i)+"_"+name_char+layer+"_Bayesian_s22.csv", np.array(temp_s22), delimiter=",")
        np.savetxt(save_dir+str(i)+"_"+name_char+layer+"_Bayesian_s33.csv", np.array(temp_s33), delimiter=",")
        np.savetxt(save_dir+str(i)+"_"+name_char+layer+"_Bayesian_s12.csv", np.array(temp_s12), delimiter=",")
        np.savetxt(save_dir+str(i)+"_"+name_char+layer+"_Bayesian_s13.csv", np.array(temp_s13), delimiter=",")
        np.savetxt(save_dir+str(i)+"_"+name_char+layer+"_Bayesian_s23.csv", np.array(temp_s23), delimiter=",")

    # Without Bayesian
    for i in range(code_results_pretrain["mean"].shape[0]):
        temp_pretrain_code_array_sampled = my_Sampler.sample(mean_vect=code_results_pretrain["mean"][i, :],
                                                            sigma_vect=code_results_pretrain["sigma"][i, :],
                                                            parser=None)

        temp_s11, temp_s22, temp_s33, temp_s12, temp_s13, temp_s23 = my_Coder.decode_array(temp_pretrain_code_array_sampled)
        np.savetxt(save_dir + str(i) + "_" + name_char + layer+"_Initial_s11.csv", np.array(temp_s11), delimiter=",")
        np.savetxt(save_dir + str(i) + "_" + name_char + layer+"_Initial_s22.csv", np.array(temp_s22), delimiter=",")
        np.savetxt(save_dir + str(i) + "_" + name_char + layer+"_Initial_s33.csv", np.array(temp_s33), delimiter=",")
        np.savetxt(save_dir + str(i) + "_" + name_char + layer+"_Initial_s12.csv", np.array(temp_s12), delimiter=",")
        np.savetxt(save_dir + str(i) + "_" + name_char + layer+"_Initial_s13.csv", np.array(temp_s13), delimiter=",")
        np.savetxt(save_dir + str(i) + "_" + name_char + layer+"_Initial_s23.csv", np.array(temp_s23), delimiter=",")


def label_force_predict():
    my_Force_Parser = ForceParser(type="BM")
    stress_vect = torch.from_numpy(np.load(r'E:\Data\DATASET\SealDigitTwin\FINAL\Fig2_STRESS\input\20.6\BM_25_0_20.6_STRESS.npy').astype("float32")).unsqueeze(0).cuda()
    force_predict = my_Force_Parser.predict_force(stress_vect)
    print(force_predict)


if __name__ == '__main__':
    label_force_predict()
    disp = 20.6
    force_label = np.loadtxt(r'E:\Data\DATASET\SealDigitTwin\FINAL\Fig2_STRESS\input\force_label.txt', delimiter='\t')
    force = [force_label[np.argmin(abs(force_label[:, 0] - disp)), 1]]
    stress_predict_guassian(init_npys_dir=r"E:\Data\DATASET\SealDigitTwin\STRESS\TRAIN\\",
                            save_dir=r"E:\Data\DATASET\SealDigitTwin\FINAL\Fig2_STRESS\results\\",
                            predict_npys_dir=r"E:\Data\DATASET\SealDigitTwin\FINAL\Fig2_STRESS\input\20.6\\",
                            name_char="20.6_",
                            force_list=force,
                            layer="BM")

