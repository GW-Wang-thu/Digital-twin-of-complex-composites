import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class Predictor(nn.Module):
    """Implementation of VAE(Variational Auto-Encoder)"""
    def __init__(self, vect_length, num_layer, in_channel=6):
        super(Predictor, self).__init__()
        self.in_channel = in_channel
        self.vect_length = vect_length
        p1 = 0.9
        p2 = 0.3
        self.dnn_1 = nn.Sequential(
            nn.Dropout(p=p1),
            nn.Linear(vect_length, vect_length // 256),
            nn.ReLU(),
        )
        self.dnn_2 = nn.Sequential(
            nn.Dropout(p=p1),
            nn.Linear(vect_length, vect_length // 256),
        )
        self.dnn_3 = nn.Sequential(
            nn.Dropout(p=p1),
            nn.Linear(vect_length, vect_length // 256),
            nn.ReLU(),
        )
        self.dnn_4 = nn.Sequential(
            nn.Dropout(p=p1),
            nn.Linear(vect_length, vect_length // 256),
            nn.ReLU(),
        )
        self.dnn_5 = nn.Sequential(
            nn.Dropout(p=p1),
            nn.Linear(vect_length, vect_length // 256),
            nn.ReLU(),
        )
        self.dnn_6 = nn.Sequential(
            nn.Dropout(p=p1),
            nn.Linear(vect_length, vect_length // 256),
            nn.ReLU(),
        )

        vect_length_stacked = 6 * (vect_length // 256)
        self.encoder = nn.Sequential(
            nn.Dropout(p=p2),
            nn.Linear(vect_length_stacked, 1),
        )

        print(self.dnn_1, self.encoder)

    def forward(self, x):
        code_1 = self.dnn_1(x[:, 0, :]).unsqueeze(1)
        code_2 = self.dnn_2(x[:, 1, :]).unsqueeze(1)
        code_3 = self.dnn_3(x[:, 2, :]).unsqueeze(1)
        code_4 = self.dnn_4(x[:, 3, :]).unsqueeze(1)
        code_5 = self.dnn_5(x[:, 4, :]).unsqueeze(1)
        code_6 = self.dnn_6(x[:, 5, :]).unsqueeze(1)

        code_all = torch.cat([code_1, code_2, code_3, code_4, code_5, code_6], dim=2)
        predicted_force = self.encoder(code_all)
        return predicted_force
