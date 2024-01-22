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
        self.autocoder.load_state_dict(torch.load(r"D:\Codes\DigitTwinRubber\STRESS\params1\BM_coder_last.pth"))
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
        # self.Force_predictor = Predictor_Force_DP(in_channel=6, vect_length=vect_length, num_layer=2).cuda()
        # checkpoint = torch.load("../../Force/params1019/" + type + "_predictor_best.pth")
        # self.Force_predictor.load_state_dict(checkpoint)
        # self.Force_predictor.eval()

    def predict_force(self, stress_value):
        force_predicted = 0.0104 * stress_value ** 2 - 0.081 * stress_value + 0.1076
        return force_predicted


class Sampler():
    def __init__(self, num_sample):
        self.num_sample = num_sample

    def sample_gaussian(self, mean_vect, sigma_vect):
        return np.random.normal(loc=mean_vect, scale=sigma_vect)

    def sample(self, mean_vect, sigma_vect, parser=None, decoder=None, force_label=None, tolerance=0.03):
        rec = []
        if parser is None:
            for i in range(self.num_sample):
                temp_sample = self.sample_gaussian(mean_vect, sigma_vect).astype("float32")
                rec.append(temp_sample)
            key = False
        else:
            i = 0
            while(len(rec) < self.num_sample and (i < 2000)):
                i += 1
                temp_sample = self.sample_gaussian(mean_vect, sigma_vect).astype("float32")
                force_predicted = parser.predict_force(decoder.decode_any(temp_sample, mode="array")[0, 1, 1839].item())
                if abs((force_predicted - force_label) / force_label) < tolerance:
                    # print(i, "\tavailable, label force: %.2f, predicted force: %.2f"%(force_label, force_predicted))
                    rec.append(temp_sample)
                # else:
                    # print("unavailable, label force: %.2f, predicted force: %.2f"%(force_label, force_predicted))
            key = i
            if i >= 1999:
                key = False
                # print("Fail in Force Filter")
                for i in range(self.num_sample):
                    temp_sample = self.sample_gaussian(mean_vect, sigma_vect).astype("float32")
                    rec.append(temp_sample)
            elif i > 100:
                print('采样循环次数： %d, 筛选率%.2f'%(i, 100/i))
        all_coord_codes = np.array(rec)
        return all_coord_codes, key

class GP():
    def __init__(self):
        self.model = None
    def train_GP(self, type, stage, train_code_predict, train_code_encoder, eval_code_predict):
        num_task = 72
        training_iter = 800

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

def stress_predict_guassian(init_npys_dir, disp_force_list, save_dir, name_char="", layer="BM"):
    '''Objects'''
    # Coder
    ratio_vect = [4.5, 1.0, 4.5, 12.0, 25.0, 8.0]
    my_Coder = Coder_VAE(type=layer, ratio_vect=ratio_vect)
    # Predictor
    my_Predictor = Predictor(code_length=72, params_length=3, hiden_length=72).cuda()
    # my_Predictor.load_state_dict(torch.load("../../STRESS/params1/"+layer+"_predictor_best.pth"))
    my_Predictor.load_state_dict(torch.load(r"D:\Codes\DigitTwinRubber\STRESS\params1\BM_predictor_best.pth"))

    my_Predictor.eval()
    # Dataloader
    my_init_Dataloader = DataLoader(PreDataloader(init_npys_dir, type=layer, ratio_vect=ratio_vect), batch_size=1, shuffle=False)
    # Sampler
    my_Sampler = Sampler(num_sample=100)
    # Force Parser
    my_Force_Parser = ForceParser(type=layer)
    my_GP_0 = GP()
    my_GP = GP()

    '''From NPY to Codes'''
    inputs_init_rec = []
    Codes_init_Coder_Rec = []
    Codes_init_Predict_Rec = []
    inputs_topre_rec = []
    Codes_topre_Predict_Rec = []

    # BM
    for i, [inputs, vects, _] in enumerate(my_init_Dataloader):
        temp_inputs = inputs.detach().cpu().numpy()
        temp_codes_encode = my_Coder.encode_any(vects)[0]
        temp_codes_predicted = my_Predictor(inputs)[0].detach().cpu().numpy()
        inputs_init_rec.append(temp_inputs)
        Codes_init_Coder_Rec.append(temp_codes_encode)
        Codes_init_Predict_Rec.append(temp_codes_predicted)

    for i in range(len(disp_force_list)):
        temp_inputs = torch.from_numpy(np.array([25, disp_force_list[i][0], disp_force_list[i][1]], dtype='float32')).cuda()
        temp_codes_predicted = my_Predictor(temp_inputs).detach().cpu().numpy()
        inputs_topre_rec.append(temp_inputs)
        Codes_topre_Predict_Rec.append(temp_codes_predicted)

    '''Train and predict using INIT data'''

    '''Sample and Decode to COORD'''
    # 用已有数据训练一下
    code_results_pretrain_0 = my_GP_0.train_GP(type="BM", stage="PreTrain",
                                             train_code_predict=np.array(Codes_init_Predict_Rec),
                                             train_code_encoder=np.array(Codes_init_Coder_Rec),
                                             eval_code_predict=np.array(Codes_topre_Predict_Rec))

    # PRETRAIN
    # Bayesian
    update_tuple = []
    for train_ord in range(len(disp_force_list)):
        temp_force = disp_force_list[train_ord][2]

        # BM
        # 如果新加入数据是新的，并且之前也没更新过，则会更新模型
        if train_ord==0:
            update_flag = True
        # 更新模型，拉掉flag
        if update_flag:
            code_results_pretrain = my_GP.train_GP(type="BM", stage="PreTrain",
                                             train_code_predict=np.array(Codes_init_Predict_Rec),
                                             train_code_encoder=np.array(Codes_init_Coder_Rec),
                                             eval_code_predict=np.array(Codes_topre_Predict_Rec))
            update_flag = False
            just_updated = True
        # 如果新数据很小，则使用原始模型，直接输出
        elif disp_force_list[train_ord][1] < 16.0:
            print("在第%d次循环不更新GP模型。"%(train_ord))
            code_results_pretrain = my_GP_0.eval_GP(np.array(Codes_topre_Predict_Rec))
            just_updated = False
        # 如果不是新数据，则使用上次更新模型输出
        else:
            print("在第%d次循环不更新GP模型。"%(train_ord))
            code_results_pretrain = my_GP.eval_GP(np.array(Codes_topre_Predict_Rec))
            just_updated = False

        for j in range(len(disp_force_list)):
            if not os.path.exists(save_dir + str(j)):
                os.mkdir(save_dir + str(j))
            if not just_updated and j != train_ord:
                continue
            temp_pretrain_code_array_sampled, key = my_Sampler.sample(mean_vect=code_results_pretrain["mean"][j, :],
                                                                 sigma_vect=code_results_pretrain["sigma"][j, :],
                                                                 parser=my_Force_Parser,
                                                                 decoder=my_Coder,
                                                                 force_label=temp_force,
                                                                 tolerance=0.05)
            # 如果是将要预测的，并且key=1（筛到了100个），并且是新数据，并且位移较大，则有新数据加入
            if j == train_ord and key > 110 and key<1999 and disp_force_list[train_ord][1] > 16.0 and (disp_force_list[train_ord][0], disp_force_list[train_ord][1]) not in update_tuple:
                print("加入新数据，将在第%d次循环更新GP模型，添加位移-力测量数据："%(train_ord+1), disp_force_list[train_ord])
                update_flag = True
                clean_bayesian_code = np.mean(temp_pretrain_code_array_sampled, axis=0, dtype="float32")
                Codes_init_Predict_Rec.append(code_results_pretrain["mean"][0].astype('float32'))           # 直接预测的做输入
                Codes_init_Coder_Rec.append(clean_bayesian_code)                                    # Bayesian筛选后的做输出
                update_tuple.append((disp_force_list[train_ord][0], disp_force_list[train_ord][1]))
            # 计算待预测的结果如果是刚更新的模型，则存储数据，否则不存储空数据
            if just_updated:
                temp_s11, temp_s22, temp_s33, temp_s12, temp_s13, temp_s23 = my_Coder.decode_array(temp_pretrain_code_array_sampled)
            else:
                temp_s11 = temp_s22 = temp_s33 = temp_s12 = temp_s13 = temp_s23 = np.zeros(shape=(1,),  dtype='uint8')
            np.save(save_dir + str(j) + "\\" + str(train_ord) + "_" + name_char + layer + "_Bayesian_s11.npy", np.array(temp_s11, dtype='float32'))
            np.save(save_dir + str(j) + "\\" + str(train_ord) + "_" + name_char + layer + "_Bayesian_s22.npy", np.array(temp_s22, dtype='float32'))
            np.save(save_dir + str(j) + "\\" + str(train_ord) + "_" + name_char + layer + "_Bayesian_s33.npy", np.array(temp_s33, dtype='float32'))
            np.save(save_dir + str(j) + "\\" + str(train_ord) + "_" + name_char + layer + "_Bayesian_s12.npy", np.array(temp_s12, dtype='float32'))
            np.save(save_dir + str(j) + "\\" + str(train_ord) + "_" + name_char + layer + "_Bayesian_s13.npy", np.array(temp_s13, dtype='float32'))
            np.save(save_dir + str(j) + "\\" + str(train_ord) + "_" + name_char + layer + "_Bayesian_s23.npy", np.array(temp_s23, dtype='float32'))



def label_force_predict():
    my_Force_Parser = ForceParser(type="BM")
    stress_vect = torch.from_numpy(np.load(r'E:\Data\DATASET\SealDigitTwin\Results\0_input_npy\STRESS\BM_25_0_21.0_STRESS.npy').astype("float32")).unsqueeze(0).cuda()
    force_predict = my_Force_Parser.predict_force(stress_vect)
    print(force_predict)


if __name__ == '__main__':
    # label_force_predict()
    force_label = np.loadtxt(r'E:\Data\DATASET\SealDigitTwin\FINAL\STRESS\input\force_label.txt', delimiter='\t')
    pressure_disp_force = np.array([100 - force_label[:, 0], force_label[:, 2], force_label[:, 3]], dtype='float32').T
    stress_predict_guassian(init_npys_dir=r"E:\Data\DATASET\SealDigitTwin\STRESS\TRAIN\\",
                            save_dir=r"E:\Data\DATASET\SealDigitTwin\FINAL\STRESS\results_6\\",
                            name_char="20221021_",
                            disp_force_list=pressure_disp_force.tolist(),
                            layer="BM")



