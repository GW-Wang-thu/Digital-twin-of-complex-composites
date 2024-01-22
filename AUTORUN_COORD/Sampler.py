import numpy as np
import torch
from STRESS.network import Predictor_LOAD


def main_coords():
    def sampler(mean_vect, sigma_vect):
        return np.random.normal(loc=mean_vect, scale=sigma_vect)
    layer = "BM"
    idx = 3
    num_sample = 100
    type = "AFTER"
    mean_vect = np.loadtxt(layer + "_train_codes_"+type+"_ADD_mean.txt", dtype="float32")[idx, :]
    lb_vect = np.loadtxt(layer + "_train_codes_"+type+"_ADD_low.txt", dtype="float32")[idx, :]
    sigma_vect = (mean_vect - lb_vect) / 2
    rec = []
    for i in range(num_sample):
        rec.append(sampler(mean_vect, sigma_vect))
    all_stress_codes = np.array(rec)
    np.savetxt(layer + "_" + str(idx)+"_"+type+"_COORD_CODES_SAMPLES.txt", all_stress_codes)


def main_stress():
    def sampler(mean_vect, sigma_vect):
        return np.random.normal(loc=mean_vect, scale=sigma_vect)
    layer = "BM"
    idx = 3
    num_sample = 100
    type = "PRE"
    mean_vect = np.loadtxt(layer + "_train_STRESS_ADD_" + type + "_mean.txt", dtype="float32")[idx, :]
    lb_vect = np.loadtxt(layer + "_train_STRESS_ADD_" + type + "_low.txt", dtype="float32")[idx, :]
    sigma_vect = (mean_vect - lb_vect) / 2
    rec = []
    for i in range(num_sample):
        rec.append(sampler(mean_vect, sigma_vect))
    all_stress_codes = np.array(rec)
    np.savetxt(layer + "_" + str(idx)+"_"+type+"_STRESS_CODES_SAMPLES.txt", all_stress_codes)

class Sampler_Force_Filter:
    def __init__(self):
        self.Load_predictor = Predictor_LOAD(code_length=72, hiden_length=72).cuda()
        self.Load_predictor.load_state_dict(torch.load("../STRESS/Load_Predictor_best.pth"))
        self.Load_predictor.eval()
        self.force_label = [0.937681913, 1.460906982, 1.166471243, 1.082090139, 1.260332346, 1.006217837]

    def sampler(self, mean_vect, sigma_vect):
        return np.random.normal(loc=mean_vect, scale=sigma_vect)

    def filter(self, stress_vect, idx):
        stress_vect = torch.from_numpy(stress_vect).unsqueeze(0).cuda()
        force = self.Load_predictor(stress_vect)
        force_label = self.force_label[idx]
        if abs((force - force_label) / force_label) < .05:
            return stress_vect[0, :].detach().cpu().numpy()
        else:
            return None

    def main(self):
        layer = "BM"
        idx = 3
        num_sample = 100
        type = "PRE"
        mean_vect = np.loadtxt(layer + "_train_STRESS_ADD_" + type + "_mean.txt", dtype="float32")[idx, :]
        lb_vect = np.loadtxt(layer + "_train_STRESS_ADD_" + type + "_low.txt", dtype="float32")[idx, :]
        sigma_vect = (mean_vect - lb_vect)/2
        rec = []
        for i in range(num_sample):
            temp_sample = self.sampler(mean_vect, sigma_vect)
            filterd_sample = self.filter(temp_sample.astype("float32"), idx)
            if filterd_sample is not None:
                print(i)
                rec.append(filterd_sample)
            else:
                print("%d is an INVALID SAMPLE"%(i))
        all_stress_codes = np.array(rec)
        np.savetxt(layer + "_" + str(idx) + "_" + type + "_STRESS_Filtered_CODES_SAMPLES.txt", all_stress_codes)


if __name__ == "__main__":
    '''option 1'''
    main_stress()

    '''option 2'''
    # my_sampler = Sampler_Force_Filter()
    # my_sampler.main()