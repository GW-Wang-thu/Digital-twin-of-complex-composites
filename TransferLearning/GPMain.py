import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
from AutoRun.STRESS.network import Predictor, VAE, Predictor_Force, Predictor_Force_DP
from AutoRun.COORD.network import VAE as VAE_COORD
from torch.utils.data import Dataset, DataLoader
import gpytorch
import time
from Continuum.C2S_Code import Continuum
import matplotlib.pyplot as plt


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
        self.autocoder.load_state_dict(torch.load(r".\params_merge\BM_coder_last.pth"))
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

        return np.array(s11_rec), np.array(s22_rec), np.array(s33_rec), np.array(s12_rec), np.array(s13_rec), np.array(s23_rec)

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


class Sampler_clean():
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


class GP():
    def __init__(self):
        self.model = None
    def train_GP(self, type, stage, train_code_predict, train_code_encoder, eval_code_predict):
        num_task = 72
        training_iter = 2000

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
        self.model = model
        predicted = model(input)
        mean = predicted.mean.detach().cpu().numpy()
        low, up = predicted.confidence_region()
        # eval_shape = (100, 100, 2)
        results = {
            "mean": mean,
            "sigma": (mean-low.detach().cpu().numpy()) / 2,
        }
        return results

    def eval_GP(self, eval_code_predict):

        eval_x = torch.from_numpy(eval_code_predict)
        input = eval_x
        predicted = self.model(input)
        mean = predicted.mean.detach().cpu().numpy()
        low, up = predicted.confidence_region()
        # eval_shape = (100, 100, 2)
        results = {
            "mean": mean,
            "sigma": (mean-low.detach().cpu().numpy()) / 2,
        }
        return results


def stress_predict_guassian(init_npys_dir,
                            to_predict_list,
                            layer="BM",
                            node_id_list=[0],
                            component=0):
    '''Objects'''
    # Coder
    ratio_vect = [4.5, 1.0, 4.5, 12.0, 25.0, 8.0]
    my_Coder_stress = Coder_VAE(type=layer, ratio_vect=ratio_vect)
    # Predictor
    my_Predictor = Predictor(code_length=72, params_length=3, hiden_length=72).cuda()
    my_Predictor.load_state_dict(torch.load(r"./params_merge/BM_predictor_best.pth"))
    my_Predictor.eval()
    # Dataloader
    my_init_Dataloader = DataLoader(PreDataloader(init_npys_dir, type=layer, ratio_vect=ratio_vect), batch_size=1, shuffle=False)
    # Sampler
    my_Sampler_clean = Sampler_clean(num_sample=100)
    # GP
    my_GP = GP()

    '''From NPY to Codes'''
    inputs_all_rec = []
    Codes_all_Coder_Rec = []
    Codes_all_Predict_Rec = []
    Codes_topre_Predict_Rec = []

    # BM
    for i, [inputs, vects, _] in enumerate(my_init_Dataloader):
        temp_inputs = inputs.detach().cpu().numpy()
        temp_codes_encode = my_Coder_stress.encode_any(vects)[0]
        temp_codes_predicted = my_Predictor(inputs)[0].detach().cpu().numpy()
        inputs_all_rec.append(temp_inputs)
        Codes_all_Coder_Rec.append(temp_codes_encode)
        Codes_all_Predict_Rec.append(temp_codes_predicted)

    for i in range(len(to_predict_list)):
        temp_inputs = torch.from_numpy(np.array(to_predict_list[i], dtype='float32')).cuda()
        temp_codes_predicted = my_Predictor(temp_inputs).detach().cpu().numpy()
        Codes_topre_Predict_Rec.append(temp_codes_predicted)

    code_results_pretrain = my_GP.train_GP(type="BM",
                                           stage="PreTrain",
                                           train_code_predict=np.array(Codes_all_Predict_Rec),
                                           train_code_encoder=np.array(Codes_all_Coder_Rec),
                                           eval_code_predict=np.array(Codes_topre_Predict_Rec))
    stress_mean_rec = []
    stress_std_rec = []
    for node_id in node_id_list:
        for j in range(len(to_predict_list)):
            temp_pretrain_code_array_sampled = my_Sampler_clean.sample(mean_vect=code_results_pretrain["mean"][j, :],
                                                                       sigma_vect=code_results_pretrain["sigma"][j, :], )
            temp_s11, temp_s22, temp_s33, temp_s12, temp_s13, temp_s23 = my_Coder_stress.decode_array(temp_pretrain_code_array_sampled)
            stress_mean_rec.append([temp_s11[:, node_id].mean(), temp_s22[:, node_id].mean(), temp_s33[:, node_id].mean(),
                                    temp_s12[:, node_id].mean(), temp_s13[:, node_id].mean(), temp_s23[:, node_id].mean()])
            stress_std_rec.append([temp_s11[:, node_id].std(), temp_s22[:, node_id].std(), temp_s33[:, node_id].std(),
                                    temp_s12[:, node_id].std(), temp_s13[:, node_id].std(), temp_s23[:, node_id].std()])
        np.savetxt(str(node_id)+"_mean.csv", np.array(stress_mean_rec), delimiter=',')
        np.savetxt(str(node_id)+"_std.csv", np.array(stress_std_rec), delimiter=',')

        mean_rec = np.array(stress_mean_rec)[:, 1].tolist()
        std_rec = np.array(stress_std_rec)[:, 1].tolist()
        plt.plot(mean_rec)
        plt.title("predicted_stress")
        plt.fill_between(range(len(np.array(stress_mean_rec)[:, 1])),
                         [mean_rec[i]-std_rec[i]*2 for i in range(len(mean_rec))],
                         [mean_rec[i]+std_rec[i]*2 for i in range(len(mean_rec))],
                         color="b",
                         alpha=0.1
                         )
        plt.show()
        stress_mean_rec = []
        stress_std_rec = []


if __name__ == '__main__':
    load_array = np.loadtxt(r'list_load.txt', delimiter='\t')
    load_list = []
    for i in range(load_array.shape[0]):
        load_list.append([load_array[i, 2], load_array[i, 3] - load_array[i, 4], load_array[i, 1]])

    stress_predict_guassian(init_npys_dir=r"E:\Data\DATASET\SealDigitTwin\TransLearn\NPY\MergeTrain\\",
                            to_predict_list=load_list,
                            layer="BM",
                            node_id_list=[1800, 1000])
