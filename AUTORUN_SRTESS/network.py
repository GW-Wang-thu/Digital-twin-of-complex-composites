import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class AUTOCODER(nn.Module):
    def __init__(self, vect_length, num_layer, code_length):
        super(AUTOCODER, self).__init__()
        ds_fact_ds = (vect_length / code_length) ** (1 / num_layer)
        encoder_layers = []
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
        print(self.encoder, self.decoder)

    def forward(self, x, coder=2):
        if self.training and coder==2:
            code = self.encoder(x)
            return self.decoder(code)
        else:
            if coder == 0:
                return self.encoder(x)
            elif coder == 1:
                return self.decoder(x)


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

        print(self.encoder, self.decoder)

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


class VAE(nn.Module):
    """Implementation of VAE(Variational Auto-Encoder)"""
    def __init__(self, vect_length, num_layer, code_length, in_channel=6):
        super(VAE, self).__init__()
        self.in_channel = in_channel
        self.vect_length = vect_length

        self.dnn_1 = nn.Sequential(
            nn.Linear(vect_length, vect_length // 8),
            nn.ReLU(),
        )
        self.dnn_2 = nn.Sequential(
            nn.Linear(vect_length, vect_length // 8),
            nn.ReLU(),
        )
        self.dnn_3 = nn.Sequential(
            nn.Linear(vect_length, vect_length // 8),
            nn.ReLU(),
        )
        self.dnn_4 = nn.Sequential(
            nn.Linear(vect_length, vect_length // 8),
            nn.ReLU(),
        )
        self.dnn_5 = nn.Sequential(
            nn.Linear(vect_length, vect_length // 8),
            nn.ReLU(),
        )
        self.dnn_6 = nn.Sequential(
            nn.Linear(vect_length, vect_length // 8),
            nn.ReLU(),
        )

        vect_length_stacked = 6 * vect_length // 8
        self.step = vect_length // 8
        ds_fact_ds = (vect_length_stacked / code_length) ** (1 / num_layer)
        encoder_layers = []

        for i in range(num_layer - 1):
            encoder_layers.append(nn.Linear(int(vect_length_stacked // ds_fact_ds ** (i)),
                                            int(vect_length_stacked // ds_fact_ds ** (i + 1))))
            encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        self.encoder_tomean = nn.Linear(int(vect_length_stacked // ds_fact_ds ** (num_layer - 1)), code_length)
        self.encoder_tosigma = nn.Linear(int(vect_length_stacked // ds_fact_ds ** (num_layer - 1)), code_length)

        ds_fact_us = (vect_length_stacked / code_length) ** (1 / num_layer)
        decoder_layers = []
        for i in range(num_layer - 1):
            decoder_layers.append(
                nn.Linear(int(code_length * ds_fact_us ** i), int(code_length * ds_fact_us ** (i + 1))))
            decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(int(code_length * ds_fact_us ** (num_layer - 1)), vect_length_stacked))
        self.decoder = nn.Sequential(*decoder_layers)


        self.dnn_u1 = nn.Sequential(
            nn.Linear(vect_length//8, vect_length),
        )
        self.dnn_u2 = nn.Sequential(
            nn.Linear(vect_length//8, vect_length),
        )
        self.dnn_u3 = nn.Sequential(
            nn.Linear(vect_length//8, vect_length),
        )
        self.dnn_u4 = nn.Sequential(
            nn.Linear(vect_length//8, vect_length),
        )
        self.dnn_u5 = nn.Sequential(
            nn.Linear(vect_length//8, vect_length),
        )
        self.dnn_u6 = nn.Sequential(
            nn.Linear(vect_length//8, vect_length),
        )

        print(self.dnn_1, self.encoder, self.decoder, self.dnn_u1)

    def encode(self, x):
        code_1 = self.dnn_1(x[:, 0, :]).unsqueeze(1)
        code_2 = self.dnn_2(x[:, 1, :]).unsqueeze(1)
        code_3 = self.dnn_3(x[:, 2, :]).unsqueeze(1)
        code_4 = self.dnn_4(x[:, 3, :]).unsqueeze(1)
        code_5 = self.dnn_5(x[:, 4, :]).unsqueeze(1)
        code_6 = self.dnn_6(x[:, 5, :]).unsqueeze(1)

        code_all = torch.cat([code_1, code_2, code_3, code_4, code_5, code_6], dim=2)
        code_raw = self.encoder(code_all)

        mean = self.encoder_tomean(code_raw)
        log_sigma = self.encoder_tosigma(code_raw)

        return mean, log_sigma

    def decode(self, code):
        decode = self.decoder(code)
        decode_1 = self.dnn_u1(decode[:, :, 0:self.step])
        decode_2 = self.dnn_u2(decode[:, :, self.step:self.step * 2])
        decode_3 = self.dnn_u3(decode[:, :, self.step * 2:self.step * 3])
        decode_4 = self.dnn_u4(decode[:, :, self.step * 3:self.step * 4])
        decode_5 = self.dnn_u5(decode[:, :, self.step * 4:self.step * 5])
        decode_6 = self.dnn_u6(decode[:, :, self.step * 5:])
        decoded_stress = torch.cat([decode_1, decode_2, decode_3, decode_4, decode_5, decode_6], dim=1)
        return decoded_stress

    def reparametrize(self, mean, log_sigma):
        std = torch.exp(log_sigma)
        eps = torch.randn_like(log_sigma)  # simple from standard normal distribution
        z = mean + eps * std
        return z

    def forward(self, x):
        mean, log_std = self.encode(x)
        z = self.reparametrize(mean, log_std)
        recon = self.decode(z)
        return recon, mean, log_std

    def loss_function(self, recon, x, mean, log_std) -> torch.Tensor:
        recon_loss = F.mse_loss(recon, x, reduction="sum")  # use "mean" may have a bad effect on gradients
        kl_loss = -0.5 * (1 + 2*log_std - mean.pow(2) - torch.exp(2*log_std))
        kl_loss = torch.sum(kl_loss)
        loss = recon_loss + kl_loss
        return loss, recon_loss


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

    def forward(self, x):
        return self.model(x)


class Predictor_Force(nn.Module):
    """Implementation of VAE(Variational Auto-Encoder)"""
    def __init__(self, vect_length, num_layer, in_channel=6):
        super(Predictor_Force, self).__init__()
        self.in_channel = in_channel
        self.vect_length = vect_length

        self.dnn_1 = nn.Sequential(
            nn.Linear(vect_length, vect_length // 256),
        )
        self.dnn_2 = nn.Sequential(
            nn.Linear(vect_length, vect_length // 256),
        )
        self.dnn_3 = nn.Sequential(
            nn.Linear(vect_length, vect_length // 256),
        )
        self.dnn_4 = nn.Sequential(
            nn.Linear(vect_length, vect_length // 256),
        )
        self.dnn_5 = nn.Sequential(
            nn.Linear(vect_length, vect_length // 256),
        )
        self.dnn_6 = nn.Sequential(
            nn.Linear(vect_length, vect_length // 256),
        )

        vect_length_stacked = 6 * (vect_length // 256)
        self.encoder = nn.Linear(vect_length_stacked, 1)

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



class Predictor_Force_DP(nn.Module):
    """Implementation of VAE(Variational Auto-Encoder)"""
    def __init__(self, vect_length, num_layer, in_channel=6):
        super(Predictor_Force_DP, self).__init__()
        self.in_channel = in_channel
        self.vect_length = vect_length
        p1 = 0
        p2 = 0
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