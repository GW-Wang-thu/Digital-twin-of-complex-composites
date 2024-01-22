import torch
import numpy as np
from COORD.network import AUTOCODER_array
from torch.utils.data import Dataset, DataLoader
import os
# from COORD_NL.train_coord import coord_predictor

class Decoder():
    def __init__(self, type):
        if type == "BM":
            vect_length = 15184
        elif type == "NB":
            vect_length = 13120
        else:
            vect_length = 26016
        self.autocoder = AUTOCODER_array(vect_length=vect_length, code_length=24, num_layer=3).cuda()
        self.autocoder.load_state_dict(torch.load("../COORD/" + type + "_coder_best.pth"))
        self.autocoder.eval()

    def decode_array(self, array):
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

    def decode_any(self, vect):
        vect = torch.from_numpy(vect).cuda()
        temp_code = vect.unsqueeze(0).unsqueeze(0)
        decoded_stress = self.autocoder(temp_code, coder=1)[0, :].detach().cpu().numpy()
        stress = decoded_stress
        return stress

    def encode_decode_any(self, stress_array):
        vect_torch = torch.from_numpy(stress_array).cuda()
        temp_code = torch.transpose(torch.reshape(vect_torch, shape=(vect_torch.shape[0] // 3, 3)), dim0=0, dim1=1).unsqueeze(0)
        code = self.autocoder(temp_code, coder=0)
        decoded_stress = self.autocoder(code, coder=1)[0, :].detach().cpu().numpy()
        stress = decoded_stress
        return stress


def Decode_array(type, idx, stage):

    codes = torch.from_numpy(np.loadtxt(type + "_" + str(idx)+"_"+stage+"_COORD_CODES_SAMPLES.txt").astype("float32")).cuda()
    my_autocoder = Decoder(type)
    x_rec, y_rec, z_rec = my_autocoder.decode_array(codes)
    np.savetxt(type + "_" + str(idx)+"_"+stage+"_X_ADD_SAMPLES.csv", np.array(x_rec, dtype="float32"), delimiter=",")
    np.savetxt(type + "_" + str(idx)+"_"+stage+"_Y_ADD_SAMPLES.csv", np.array(y_rec, dtype="float32"), delimiter=",")
    np.savetxt(type + "_" + str(idx)+"_"+stage+"_Z_ADD_SAMPLES.csv", np.array(z_rec, dtype="float32"), delimiter=",")


def Decode_vect(type="BM"):
    vect = np.loadtxt("E:\Data\DATASET\SealDigitTwin\Results\\1.5_input_stress_code_PR&AC\\BM_train_STRESS_PRE_ADD.txt", dtype="float32")[0, 3:]
    my_autocoder = Decoder(type)
    stress = my_autocoder.decode_any(vect[:72])
    np.savetxt(type + "_TEMP_STRESS.csv", stress, delimiter=",")


def Encode_Decode_vect(type="BM"):
    vect = np.load("E:\Data\DATASET\SealDigitTwin\Results\\0_input_npy\COORD\\BM_25_5_20.0_Coord.npy").astype("float32")
    my_autocoder = Decoder(type)
    stress = my_autocoder.encode_decode_any(vect)
    np.savetxt(type + "_TEMP_COORD.csv", stress, delimiter=",")


if __name__ == '__main__':
    type="BM"
    idx="3"
    stage="AFTER"
    # Decode_array(type, idx, stage)

    # Decode_vect(type)

    # type = "BM"
    Encode_Decode_vect(type)

