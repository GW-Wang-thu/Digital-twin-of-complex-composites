import torch
import torch.nn as nn
import math


class AUTOCODER_array(nn.Module):
    def __init__(self, vect_length, num_layer, code_length):
        super(AUTOCODER_array, self).__init__()
        ds_fact_ds = (vect_length / code_length) ** (1 / num_layer)
        encoder_layers = []
        self.code_length = code_length
        for i in range(num_layer-1):
            encoder_layers.append(nn.Linear(int(vect_length//ds_fact_ds**(i)), int(vect_length//ds_fact_ds**(i+1))))
            encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Linear(int(vect_length//ds_fact_ds**(num_layer-1)), code_length))
        self.encoder = nn.Sequential(*encoder_layers)

        self.encoder_x = self.encoder
        self.encoder_y = self.encoder
        self.encoder_z = self.encoder

        self.encoder_all = nn.Sequential(
            nn.ReLU(),
            nn.Linear(code_length*3, code_length)
        )

        self.decoder_all = nn.Sequential(
            nn.Linear(code_length, code_length*3),
            nn.ReLU()
        )

        ds_fact_us = (vect_length / code_length) ** (1 / num_layer)
        decoder_layers = []
        for i in range(num_layer - 1):
            decoder_layers.append(nn.Linear(int(code_length * ds_fact_us ** i), int(code_length * ds_fact_us ** (i + 1))))
            decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(int(code_length * ds_fact_us ** (num_layer-1)), vect_length))
        self.decoder = nn.Sequential(*decoder_layers)
        self.decoder_x = self.decoder
        self.decoder_y = self.decoder
        self.decoder_z = self.decoder

        print(self.encoder)
        print("*********************************************************************")
        print(self.decoder)
        print("*********************************************************************")
        print(self.encoder_all)
        print("*********************************************************************")
        print(self.decoder_all)

    def forward(self, x, coder=2):
        if self.training and coder == 2:
            x_vect = x[:, 0, :]
            y_vect = x[:, 1, :]
            z_vect = x[:, 2, :]

            x_code = self.encoder_x(x_vect)
            y_code = self.encoder_y(y_vect)
            z_code = self.encoder_z(z_vect)

            code_all = torch.cat([x_code, y_code, z_code], dim=-1)
            code_out = self.encoder_all(code_all)

            decode_all = self.decoder_all(code_out)
            x_decode = self.decoder_x(decode_all[:, 0:self.code_length])
            y_decode = self.decoder_y(decode_all[:, self.code_length:self.code_length*2])
            z_decode = self.decoder_z(decode_all[:, self.code_length*2:])

            coord_array = torch.stack([x_decode, y_decode, z_decode], dim=1)
            return coord_array
        else:
            if coder == 0:
                x_vect = x[:, 0, :]
                y_vect = x[:, 1, :]
                z_vect = x[:, 2, :]

                x_code = self.encoder_x(x_vect)
                y_code = self.encoder_y(y_vect)
                z_code = self.encoder_z(z_vect)

                code_all = torch.cat([x_code, y_code, z_code], dim=-1)
                code_out = self.encoder_all(code_all)
                return code_out
            elif coder == 1:
                decode_all = self.decoder_all(x)
                x_decode = self.decoder_x(decode_all[:, 0:self.code_length])
                y_decode = self.decoder_y(decode_all[:, self.code_length:self.code_length * 2])
                z_decode = self.decoder_z(decode_all[:, self.code_length * 2:])

                coord_array = torch.stack([x_decode, y_decode, z_decode], dim=1)
                return coord_array




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



class AUTOCODER_conv(nn.Module):
    def __init__(self, vect_length, num_layer, code_length):
        super(AUTOCODER_conv, self).__init__()

        vect_length_i = vect_length
        vect_length = vect_length_i // 2
        ds_fact_ds = (vect_length / code_length) ** (1 / num_layer)

        encoder_layers = [nn.Conv1d(in_channels=1, out_channels=1, padding=1, kernel_size=3, stride=2)]
        for i in range(num_layer-1):
            encoder_layers.append(nn.Linear(int(vect_length//ds_fact_ds**(i)), int(vect_length//ds_fact_ds**(i+1))))
            encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Linear(int(vect_length//ds_fact_ds**(num_layer-1)), code_length))
        self.encoder = nn.Sequential(*encoder_layers)

        ds_fact_us = (vect_length / code_length) ** (1 / num_layer)
        decoder_layers = []
        for i in range(num_layer - 1):
            decoder_layers.append(nn.Linear(int(code_length * ds_fact_us ** i), int(code_length * ds_fact_us ** (i + 1))))
            decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(int(code_length * ds_fact_us ** (num_layer-1)), vect_length))
        self.decoder = nn.Sequential(*decoder_layers)
        print(self.encoder)
        print("*********************************************************************")
        print(self.decoder)

    def forward(self, x, coder=0):
        if self.training:
            code = self.encoder(x)
            return self.decoder(code)
        else:
            if coder == 0:
                return self.encoder(x)
            elif coder == 1:
                return self.decoder(x)



class Predictor_conv(nn.Module):
    def __init__(self, params_length, code_length, hiden_length):
        super(Predictor_conv, self).__init__()
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