import torch
import torch.nn as nn
import math


class Predictor_LOAD(nn.Module):
    def __init__(self, code_length, hiden_length):
        super(Predictor_LOAD, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(code_length, hiden_length),
            nn.ReLU(),
            nn.Linear(hiden_length, hiden_length),
            nn.ReLU(),
            nn.Linear(hiden_length, hiden_length),
            nn.ReLU(),
            nn.Linear(hiden_length, 1),
        )
        print(self.model)

    def forward(self, x):
        return self.model(x)


class Predictor(nn.Module):
    def __init__(self, params_length, code_length, hiden_length):
        super(Predictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(params_length, hiden_length),
            nn.ReLU(),
            nn.Linear(hiden_length, hiden_length),
            nn.ReLU(),
            nn.Linear(hiden_length, hiden_length),
            nn.ReLU(),
            nn.Linear(hiden_length, hiden_length),
            nn.ReLU(),
            nn.Linear(hiden_length, hiden_length),
            nn.ReLU(),
            nn.Linear(hiden_length, code_length),
        )
        print(self.model)

    def forward(self, x):
        return self.model(x)


class blocked_AUTOCODER(nn.Module):
    def __init__(self, vect_length, num_layer=3, code_length=72, in_channel=6):
        super(blocked_AUTOCODER, self).__init__()
        self.in_channel = in_channel
        self.vect_length = vect_length

        self.dnn_1 = nn.Sequential(
            nn.Linear(vect_length, vect_length//4),
            nn.ReLU(),
            nn.Linear(vect_length//4, vect_length//8),
            nn.ReLU(),
        )
        self.dnn_2 = nn.Sequential(
            nn.Linear(vect_length, vect_length // 4),
            nn.ReLU(),
            nn.Linear(vect_length // 4, vect_length // 8),
            nn.ReLU(),
        )
        self.dnn_3 = nn.Sequential(
            nn.Linear(vect_length, vect_length // 4),
            nn.ReLU(),
            nn.Linear(vect_length // 4, vect_length // 8),
            nn.ReLU(),
        )
        self.dnn_4 = nn.Sequential(
            nn.Linear(vect_length, vect_length // 4),
            nn.ReLU(),
            nn.Linear(vect_length // 4, vect_length // 8),
            nn.ReLU(),
        )
        self.dnn_5 = nn.Sequential(
            nn.Linear(vect_length, vect_length // 4),
            nn.ReLU(),
            nn.Linear(vect_length // 4, vect_length // 8),
            nn.ReLU(),
        )
        self.dnn_6 = nn.Sequential(
            nn.Linear(vect_length, vect_length // 4),
            nn.ReLU(),
            nn.Linear(vect_length // 4, vect_length // 8),
            nn.ReLU(),
        )

        vect_length_stacked = 6 * vect_length // 8
        self.step = vect_length // 8
        ds_fact_ds = (vect_length_stacked / code_length) ** (1 / num_layer)
        encoder_layers = []
        for i in range(num_layer-1):
            encoder_layers.append(nn.Linear(int(vect_length_stacked//ds_fact_ds**(i)), int(vect_length_stacked//ds_fact_ds**(i+1))))
            encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Linear(int(vect_length_stacked//ds_fact_ds**(num_layer-1)), code_length))
        self.encoder = nn.Sequential(*encoder_layers)

        ds_fact_us = (vect_length_stacked / code_length) ** (1 / num_layer)
        decoder_layers = []
        for i in range(num_layer - 1):
            decoder_layers.append(nn.Linear(int(code_length * ds_fact_us ** i), int(code_length * ds_fact_us ** (i + 1))))
            decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(int(code_length * ds_fact_us ** (num_layer-1)), vect_length_stacked))
        self.decoder = nn.Sequential(*decoder_layers)

        self.dnn_u1 = nn.Sequential(
            nn.Linear(vect_length//8, vect_length//4),
            nn.ReLU(),
            nn.Linear(vect_length//4, vect_length),
        )
        self.dnn_u2 = nn.Sequential(
            nn.Linear(vect_length//8, vect_length//4),
            nn.ReLU(),
            nn.Linear(vect_length//4, vect_length),
        )
        self.dnn_u3 = nn.Sequential(
            nn.Linear(vect_length//8, vect_length//4),
            nn.ReLU(),
            nn.Linear(vect_length//4, vect_length),
        )
        self.dnn_u4 = nn.Sequential(
            nn.Linear(vect_length//8, vect_length//4),
            nn.ReLU(),
            nn.Linear(vect_length//4, vect_length),
        )
        self.dnn_u5 = nn.Sequential(
            nn.Linear(vect_length//8, vect_length//4),
            nn.ReLU(),
            nn.Linear(vect_length//4, vect_length),
        )
        self.dnn_u6 = nn.Sequential(
            nn.Linear(vect_length//8, vect_length//4),
            nn.ReLU(),
            nn.Linear(vect_length//4, vect_length),
        )
        print(self.dnn_1, self.encoder, self.decoder, self.dnn_u1)

    def forward(self, x, coder=0):
        if self.training and coder != 1 and coder != 2:
            code_1 = self.dnn_1(x[:, 0, :]).unsqueeze(1)
            code_2 = self.dnn_2(x[:, 1, :]).unsqueeze(1)
            code_3 = self.dnn_3(x[:, 2, :]).unsqueeze(1)
            code_4 = self.dnn_4(x[:, 3, :]).unsqueeze(1)
            code_5 = self.dnn_5(x[:, 4, :]).unsqueeze(1)
            code_6 = self.dnn_6(x[:, 5, :]).unsqueeze(1)
            code_all = torch.cat([code_1, code_2, code_3, code_4, code_5, code_6], dim=2)
            code = self.encoder(code_all)

            decode = self.decoder(code)
            decode_1 = self.dnn_u1(decode[:, :, 0:self.step])
            decode_2 = self.dnn_u2(decode[:, :, self.step:self.step*2])
            decode_3 = self.dnn_u3(decode[:, :, self.step*2:self.step*3])
            decode_4 = self.dnn_u4(decode[:, :, self.step*3:self.step*4])
            decode_5 = self.dnn_u5(decode[:, :, self.step*4:self.step*5])
            decode_6 = self.dnn_u6(decode[:, :, self.step*5:])
            decoded_stress = torch.cat([decode_1, decode_2, decode_3, decode_4, decode_5, decode_6], dim=1)
            return decoded_stress
        else:
            if coder == 0:
                code_1 = self.dnn_1(x[:, 0, :]).unsqueeze(1)
                code_2 = self.dnn_2(x[:, 1, :]).unsqueeze(1)
                code_3 = self.dnn_3(x[:, 2, :]).unsqueeze(1)
                code_4 = self.dnn_4(x[:, 3, :]).unsqueeze(1)
                code_5 = self.dnn_5(x[:, 4, :]).unsqueeze(1)
                code_6 = self.dnn_6(x[:, 5, :]).unsqueeze(1)
                code_all = torch.cat([code_1, code_2, code_3, code_4, code_5, code_6], dim=2)
                code = self.encoder(code_all)
                return code

            elif coder == 2:
                decode = self.decoder(x)
                decode_1 = self.dnn_u1(decode[:, :, 0:self.step])
                decode_2 = self.dnn_u2(decode[:, :, self.step:self.step * 2])
                decode_3 = self.dnn_u3(decode[:, :, self.step * 2:self.step * 3])
                decode_4 = self.dnn_u4(decode[:, :, self.step * 3:self.step * 4])
                decode_5 = self.dnn_u5(decode[:, :, self.step * 4:self.step * 5])
                decode_6 = self.dnn_u6(decode[:, :, self.step * 5:])
                decoded_stress = torch.cat([decode_1, decode_2, decode_3, decode_4, decode_5, decode_6], dim=1)
                return decoded_stress


class blocked_AUTOCODER_pr(nn.Module):
    def __init__(self, vect_length, num_layer=3, code_length=72, in_channel=6):
        super(blocked_AUTOCODER_pr, self).__init__()
        self.in_channel = in_channel
        self.vect_length = vect_length

        self.dnn_1 = nn.Sequential(
            nn.Linear(vect_length, vect_length//8),
            nn.ReLU(),
            nn.Linear(vect_length//8, vect_length//16),
            nn.ReLU(),
        )
        self.dnn_2 = nn.Sequential(
            nn.Linear(vect_length, vect_length // 8),
            nn.ReLU(),
            nn.Linear(vect_length // 8, vect_length // 16),
            nn.ReLU(),
        )
        self.dnn_3 = nn.Sequential(
            nn.Linear(vect_length, vect_length // 8),
            nn.ReLU(),
            nn.Linear(vect_length // 8, vect_length // 16),
            nn.ReLU(),
        )
        self.dnn_4 = nn.Sequential(
            nn.Linear(vect_length, vect_length // 8),
            nn.ReLU(),
            nn.Linear(vect_length // 8, vect_length // 16),
            nn.ReLU(),
        )
        self.dnn_5 = nn.Sequential(
            nn.Linear(vect_length, vect_length // 8),
            nn.ReLU(),
            nn.Linear(vect_length // 8, vect_length // 16),
            nn.ReLU(),
        )
        self.dnn_6 = nn.Sequential(
            nn.Linear(vect_length, vect_length // 8),
            nn.ReLU(),
            nn.Linear(vect_length // 8, vect_length // 16),
            nn.ReLU(),
        )

        vect_length_stacked = 6 * vect_length // 16
        self.step = vect_length // 16
        ds_fact_ds = (vect_length_stacked / code_length) ** (1 / num_layer)
        encoder_layers = []
        for i in range(num_layer-1):
            encoder_layers.append(nn.Linear(int(vect_length_stacked//ds_fact_ds**(i)), int(vect_length_stacked//ds_fact_ds**(i+1))))
            encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Linear(int(vect_length_stacked//ds_fact_ds**(num_layer-1)), code_length))
        self.encoder = nn.Sequential(*encoder_layers)

        ds_fact_us = (vect_length_stacked / code_length) ** (1 / num_layer)
        decoder_layers = []
        for i in range(num_layer - 1):
            decoder_layers.append(nn.Linear(int(code_length * ds_fact_us ** i), int(code_length * ds_fact_us ** (i + 1))))
            decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(int(code_length * ds_fact_us ** (num_layer-1)), vect_length_stacked))
        self.decoder = nn.Sequential(*decoder_layers)

        self.dnn_u1 = nn.Sequential(
            nn.Linear(vect_length//16, vect_length//8),
            nn.ReLU(),
            nn.Linear(vect_length//8, vect_length),
        )
        self.dnn_u2 = nn.Sequential(
            nn.Linear(vect_length//16, vect_length//8),
            nn.ReLU(),
            nn.Linear(vect_length//8, vect_length),
        )
        self.dnn_u3 = nn.Sequential(
            nn.Linear(vect_length//16, vect_length//8),
            nn.ReLU(),
            nn.Linear(vect_length//8, vect_length),
        )
        self.dnn_u4 = nn.Sequential(
            nn.Linear(vect_length//16, vect_length//8),
            nn.ReLU(),
            nn.Linear(vect_length//8, vect_length),
        )
        self.dnn_u5 = nn.Sequential(
            nn.Linear(vect_length//16, vect_length//8),
            nn.ReLU(),
            nn.Linear(vect_length//8, vect_length),
        )
        self.dnn_u6 = nn.Sequential(
            nn.Linear(vect_length//16, vect_length//8),
            nn.ReLU(),
            nn.Linear(vect_length//8, vect_length),
        )
        print(self.dnn_1, self.encoder, self.decoder, self.dnn_u1)

    def forward(self, x, coder=0):
        if self.training and coder != 1 and coder != 2:
            code_1 = self.dnn_1(x[:, 0, :]).unsqueeze(1)
            code_2 = self.dnn_2(x[:, 1, :]).unsqueeze(1)
            code_3 = self.dnn_3(x[:, 2, :]).unsqueeze(1)
            code_4 = self.dnn_4(x[:, 3, :]).unsqueeze(1)
            code_5 = self.dnn_5(x[:, 4, :]).unsqueeze(1)
            code_6 = self.dnn_6(x[:, 5, :]).unsqueeze(1)
            code_all = torch.cat([code_1, code_2, code_3, code_4, code_5, code_6], dim=2)
            code = self.encoder(code_all)

            decode = self.decoder(code)
            decode_1 = self.dnn_u1(decode[:, :, 0:self.step])
            decode_2 = self.dnn_u2(decode[:, :, self.step:self.step*2])
            decode_3 = self.dnn_u3(decode[:, :, self.step*2:self.step*3])
            decode_4 = self.dnn_u4(decode[:, :, self.step*3:self.step*4])
            decode_5 = self.dnn_u5(decode[:, :, self.step*4:self.step*5])
            decode_6 = self.dnn_u6(decode[:, :, self.step*5:])
            decoded_stress = torch.cat([decode_1, decode_2, decode_3, decode_4, decode_5, decode_6], dim=1)
            return decoded_stress
        else:
            if coder == 0:
                code_1 = self.dnn_1(x[:, 0, :]).unsqueeze(1)
                code_2 = self.dnn_2(x[:, 1, :]).unsqueeze(1)
                code_3 = self.dnn_3(x[:, 2, :]).unsqueeze(1)
                code_4 = self.dnn_4(x[:, 3, :]).unsqueeze(1)
                code_5 = self.dnn_5(x[:, 4, :]).unsqueeze(1)
                code_6 = self.dnn_6(x[:, 5, :]).unsqueeze(1)
                code_all = torch.cat([code_1, code_2, code_3, code_4, code_5, code_6], dim=2)
                code = self.encoder(code_all)
                return code

            elif coder == 2:
                decode = self.decoder(x)
                decode_1 = self.dnn_u1(decode[:, :, 0:self.step])
                decode_2 = self.dnn_u2(decode[:, :, self.step:self.step * 2])
                decode_3 = self.dnn_u3(decode[:, :, self.step * 2:self.step * 3])
                decode_4 = self.dnn_u4(decode[:, :, self.step * 3:self.step * 4])
                decode_5 = self.dnn_u5(decode[:, :, self.step * 4:self.step * 5])
                decode_6 = self.dnn_u6(decode[:, :, self.step * 5:])
                decoded_stress = torch.cat([decode_1, decode_2, decode_3, decode_4, decode_5, decode_6], dim=1)
                return decoded_stress

class AUTOCODER_su(nn.Module):
    def __init__(self, code_length):
        super(AUTOCODER_su, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(code_length, code_length),
                                     nn.ReLU(),
                                     nn.Linear(code_length, code_length))

        self.decoder = nn.Sequential(nn.Linear(code_length, code_length),
                                     nn.ReLU())

    def forward(self, x, coder=0):
        if coder == 0:
            return self.decoder(self.encoder(x))
        elif coder == 1:
            return self.decoder(x)
        else:
            return self.encoder(x)

