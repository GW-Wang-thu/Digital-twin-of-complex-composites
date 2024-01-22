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
    def __init__(self, vect_length, num_layer, code_length):
        super(VAE, self).__init__()

        ds_fact_ds = (vect_length / code_length) ** (1 / num_layer)
        encoder_layers = []
        self.code_length = code_length
        for i in range(num_layer - 1):
            encoder_layers.append(
                nn.Linear(int(vect_length // ds_fact_ds ** (i)), int(vect_length // ds_fact_ds ** (i + 1))))
            encoder_layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*encoder_layers)

        self.encoder_x = self.encoder
        self.encoder_y = self.encoder
        self.encoder_z = self.encoder

        self.encoder_tomean = nn.Linear(int(vect_length//ds_fact_ds**(num_layer-1))*3, code_length)
        self.encoder_tosigma = nn.Linear(int(vect_length//ds_fact_ds**(num_layer-1))*3, code_length)

        self.decoder_all = nn.Sequential(
            nn.Linear(code_length, code_length * 3),
            nn.ReLU()
        )

        ds_fact_us = (vect_length / code_length) ** (1 / num_layer)
        decoder_layers = []
        for i in range(num_layer - 1):
            decoder_layers.append(
                nn.Linear(int(code_length * ds_fact_us ** i), int(code_length * ds_fact_us ** (i + 1))))
            decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(int(code_length * ds_fact_us ** (num_layer - 1)), vect_length))
        self.decoder = nn.Sequential(*decoder_layers)
        self.decoder_x = self.decoder
        self.decoder_y = self.decoder
        self.decoder_z = self.decoder

        print(self.encoder, self.decoder)

    def encode(self, x):
        x_vect = x[:, 0, :]
        y_vect = x[:, 1, :]
        z_vect = x[:, 2, :]

        x_code = self.encoder_x(x_vect)
        y_code = self.encoder_y(y_vect)
        z_code = self.encoder_z(z_vect)

        code_all = torch.cat([x_code, y_code, z_code], dim=-1)
        mean = self.encoder_tomean(code_all)
        log_sigma = self.encoder_tosigma(code_all)
        return mean, log_sigma

    def decode(self, code):
        decode_all = self.decoder_all(code)
        x_decode = self.decoder_x(decode_all[:, 0:self.code_length])
        y_decode = self.decoder_y(decode_all[:, self.code_length:self.code_length * 2])
        z_decode = self.decoder_z(decode_all[:, self.code_length * 2:])
        coord_array = torch.stack([x_decode, y_decode, z_decode], dim=1)
        return coord_array

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