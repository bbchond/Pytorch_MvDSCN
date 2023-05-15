import torch
import torch.nn as nn


class DeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeConvBlock, self).__init__()
        self.de_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        dec = self.de_conv(inputs)
        dec = self.relu(dec)
        return dec


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        conv = self.conv(inputs)
        conv = self.relu(conv)
        return conv


class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Encoder, self).__init__()
        self.conv_block1 = ConvBlock(in_channels=input_dim, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=64)
        self.conv_block3 = ConvBlock(in_channels=64, out_channels=output_dim)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        return x


class EncoderSingle(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EncoderSingle, self).__init__()
        self.conv_block1 = ConvBlock(in_channels=input_dim, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=64)
        self.conv_block3 = ConvBlock(in_channels=64, out_channels=output_dim)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Decoder, self).__init__()
        self.de_conv1 = DeConvBlock(in_channels=input_dim, out_channels=64)
        self.de_conv2 = DeConvBlock(in_channels=64, out_channels=64)
        # self.de_conv3 = nn.ConvTranspose2d(in_channels=64, out_channels=output_dim, kernel_size=3, stride=2,
        # output_padding=1, padding=1)
        self.de_conv3 = nn.ConvTranspose2d(in_channels=64, out_channels=output_dim, kernel_size=3, stride=2,
                                           output_padding=1, padding=1)

    def forward(self, x):
        x = self.de_conv1(x)
        x = self.de_conv2(x)
        x = self.de_conv3(x)
        return x


class DecoderSingle(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DecoderSingle, self).__init__()
        self.de_conv1 = DeConvBlock(in_channels=input_dim, out_channels=64)
        self.de_conv2 = DeConvBlock(in_channels=64, out_channels=64)
        self.de_conv3 = nn.ConvTranspose2d(in_channels=64, out_channels=output_dim, kernel_size=3, stride=2,
                                           output_padding=1, padding=1)

    def forward(self, x):
        x = self.de_conv1(x)
        x = self.de_conv2(x)
        x = self.de_conv3(x)
        return x


class SelfExpression(nn.Module):
    def __init__(self, n_samples):
        super(SelfExpression, self).__init__()
        self.cof = nn.Parameter(1.0e-8 * torch.ones(n_samples, n_samples, dtype=torch.float32), requires_grad=True)
        self.n_samples = n_samples

    def forward(self, x):
        y = torch.matmul(self.cof - torch.diag(torch.diag(self.cof)), x.view(self.n_samples, -1))
        y = y.view(x.size())
        # 返回自表示系数矩阵，以及重构的latent
        return self.cof, y


class AutoEncoderInit(nn.Module):
    def __init__(self, batch_size, ft=False):
        super(AutoEncoderInit, self).__init__()
        self.ft = ft
        # different view feature input
        self.batch_size = batch_size
        self.iter = 0

        self.encoder1 = Encoder(input_dim=3, output_dim=64)
        self.encoder2 = Encoder(input_dim=1, output_dim=64)
        self.encoder1_single = Encoder(input_dim=3, output_dim=64)
        self.encoder2_single = Encoder(input_dim=1, output_dim=64)

        self.decoder1 = Decoder(input_dim=64, output_dim=3)
        self.decoder2 = Decoder(input_dim=64, output_dim=1)
        self.decoder1_single = Decoder(input_dim=64, output_dim=3)
        self.decoder2_single = Decoder(input_dim=64, output_dim=1)
        self.self_express_view_1 = SelfExpression(batch_size)
        self.self_express_view_2 = SelfExpression(batch_size)
        self.self_express_view_common = torch.nn.Parameter\
            (1.0e-8 * torch.ones(self.batch_size, self.batch_size, dtype=torch.float32).cuda(), requires_grad=True)

    def forward(self, all_views_data):
        latent1 = self.encoder1(all_views_data[0])
        latent2 = self.encoder2(all_views_data[1])
        diversity_latent_1 = self.encoder1_single(all_views_data[0])
        diversity_latent_2 = self.encoder2_single(all_views_data[1])
        # if self.ft is True, we reconstruct data by using after self-expressive, or use without self-expressive
        if self.ft:
            # Self Expressive Layer Parts
            # Diversity Self Expressive, \|F_i^s - F_i^s * Z_i\|
            # latent1_diversity_se = torch.reshape(latent1_diversity_se, shape=diversity_latent_1.size())
            # latent2_diversity_se = torch.reshape(latent2_diversity_se, shape=diversity_latent_2.size())
            # Common Self Expressive, \|F_i^c - F_i^c * Z\|
            z1, latent1_diversity_se = self.self_express_view_1(diversity_latent_1)
            z2, latent2_diversity_se = self.self_express_view_2(diversity_latent_2)
            # Common Self Expressive Coef, \|F_i^c - F_i^c * Z_{common}\|
            z_common = self.self_express_view_common - torch.diag(torch.diag(self.self_express_view_common))
            latent1_se = torch.matmul(z_common, latent1.view(self.batch_size, -1))
            latent2_se = torch.matmul(z_common, latent2.view(self.batch_size, -1))
            latent1_se = torch.reshape(latent1_se, shape=latent1.size())
            latent2_se = torch.reshape(latent2_se, shape=latent2.size())

            view1_r = self.decoder1(latent1_se)
            view2_r = self.decoder2(latent2_se)
            view1_r_diversity = self.decoder1_single(latent1_diversity_se)
            view2_r_diversity = self.decoder2_single(latent2_diversity_se)
        else:
            view1_r = self.decoder1(latent1)
            view2_r = self.decoder2(latent2)
            view1_r_diversity = self.decoder1_single(diversity_latent_1)
            view2_r_diversity = self.decoder2_single(diversity_latent_2)

        # print(latent1.shape, view1_r.shape, view1_r_diversity.shape)
        # print(latent2.shape, view2_r.shape, view2_r_diversity.shape)

        if self.ft:
            return view1_r, view2_r, view1_r_diversity, view2_r_diversity, z1, z2, z_common, diversity_latent_1, \
                   diversity_latent_2, latent1, latent2, latent1_diversity_se, latent2_diversity_se, latent1_se, latent2_se
        else:
            return view1_r, view2_r, view1_r_diversity, view2_r_diversity
