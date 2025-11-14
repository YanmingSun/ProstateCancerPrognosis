import torch
import torch.nn as nn
import torch.nn.functional as F


'''
class MyLayerNorm3D(nn.Module):
    def __init__(self, num_channels):
        super(MyLayerNorm3D, self).__init__()
        self.num_channels = num_channels
        self.re_scale = nn.Conv3d(in_channels=num_channels, out_channels=num_channels, kernel_size=(1, 1, 1), padding='same', groups=num_channels, bias=True, padding_mode='reflect')

    def forward(self, x):
        size_x = x.size()
        total_pixel = size_x[1] * size_x[2] * size_x[3] * size_x[4]  # self.num_channels * torch.sum(mask, dim=(1, 2, 3, 4), keepdim=True)
        mu_s = torch.div(torch.sum(x, dim=(1, 2, 3, 4), keepdim=True), (total_pixel + 1e-6))
        std_s = torch.sqrt(1e-6 + torch.div(torch.sum((x - mu_s) ** 2, dim=(1, 2, 3, 4), keepdim=True), (total_pixel + 1e-6)))

        x_norm = torch.div((x - mu_s), std_s)
        x_out = self.re_scale(x_norm)

        return x_out
'''


class ClinicalEncoder(nn.Module):
    def __init__(self, in_ft, out_ft):
        super(ClinicalEncoder, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_ft, out_ft),
            nn.LayerNorm(out_ft),
            nn.ReLU(),
            nn.Linear(out_ft, out_ft),
            nn.LayerNorm(out_ft),
            nn.ReLU(),
            nn.Linear(out_ft, out_ft))

    def forward(self, x):

        return self.mlp(x)


class MRIt2wEncoder(nn.Module):
    def __init__(self, num_in_channels, num_channels, num_out_features):
        super(MRIt2wEncoder, self).__init__()

        self.conv_pool1 = nn.Sequential(
            nn.Conv3d(in_channels=num_in_channels, out_channels=int(num_channels / 2), kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1, bias=True, padding_mode='reflect'),
            nn.InstanceNorm3d(int(num_channels / 2), affine=True),
            nn.ReLU())
        self.conv_pool2 = nn.Sequential(
            nn.Conv3d(in_channels=int(num_channels / 2), out_channels=int(num_channels / 2), kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1, bias=True, padding_mode='reflect'),
            nn.InstanceNorm3d(int(num_channels / 2), affine=True),
            nn.ReLU())
        self.conv_pool3 = nn.Sequential(
            nn.Conv3d(in_channels=int(num_channels / 2), out_channels=int(num_channels / 2), kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1, bias=True, padding_mode='reflect'),
            nn.InstanceNorm3d(int(num_channels / 2), affine=True),
            nn.ReLU())
        self.conv_pool4 = nn.Sequential(
            nn.Conv3d(in_channels=int(num_channels / 2), out_channels=num_channels, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1, bias=True, padding_mode='reflect'),
            nn.InstanceNorm3d(num_channels, affine=True),
            nn.ReLU())
        self.conv_pool5 = nn.Sequential(
            nn.Conv3d(in_channels=num_channels, out_channels=num_channels, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1, bias=True, padding_mode='reflect'),
            nn.InstanceNorm3d(num_channels, affine=True),
            nn.ReLU())
        self.conv_pool6 = nn.Conv3d(in_channels=num_channels, out_channels=1, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1, bias=True, padding_mode='reflect')

        self.flat = nn.Flatten()

        self.mlp = nn.Sequential(
            nn.Linear(int(1 * 13 * 10 * 10), num_out_features * 2),
            nn.LayerNorm(num_out_features * 2),
            nn.ReLU(),
            nn.Linear(num_out_features * 2, num_out_features))

    def forward(self, x):
        x_pool1 = self.conv_pool1(x)
        x_pool2 = self.conv_pool2(x_pool1)
        x_pool3 = self.conv_pool3(x_pool2)
        x_pool4 = self.conv_pool4(x_pool3)
        x_pool5 = self.conv_pool5(x_pool4)
        x_pool6 = self.conv_pool6(x_pool5)

        x_f = self.flat(x_pool6)
        x_mlp = self.mlp(x_f)

        return x_mlp


class MRIhbvadcEncoder(nn.Module):
    def __init__(self, num_in_channels, num_channels, num_out_features):
        super(MRIhbvadcEncoder, self).__init__()

        self.conv_pool1 = nn.Sequential(
            nn.Conv3d(in_channels=num_in_channels, out_channels=num_channels, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1, bias=True, padding_mode='reflect'),
            nn.InstanceNorm3d(num_channels, affine=True),
            nn.ReLU())
        self.conv_pool2 = nn.Sequential(
            nn.Conv3d(in_channels=num_channels, out_channels=num_channels, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1, bias=True, padding_mode='reflect'),
            nn.InstanceNorm3d(num_channels, affine=True),
            nn.ReLU())
        self.conv_pool3 = nn.Conv3d(in_channels=num_channels, out_channels=1, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1, bias=True, padding_mode='reflect')

        self.flat = nn.Flatten()

        self.mlp = nn.Sequential(
            nn.Linear(int(1 * 13 * 15 * 16), num_out_features * 2),
            nn.LayerNorm(num_out_features * 2),
            nn.ReLU(),
            nn.Linear(num_out_features * 2, num_out_features))

    def forward(self, x):
        x_pool1 = self.conv_pool1(x)
        x_pool2 = self.conv_pool2(x_pool1)
        x_pool3 = self.conv_pool3(x_pool2)

        x_f = self.flat(x_pool3)
        x_mlp = self.mlp(x_f)

        return x_mlp


class MultimodalNetwork(nn.Module):
    def __init__(self):
        super(MultimodalNetwork, self).__init__()
        self.encoder_numeric_feature = ClinicalEncoder(3, 64)
        self.encoder_ordinal_feature = ClinicalEncoder(5, 64)
        self.encoder_binary_feature = ClinicalEncoder(5, 64)
        self.encoder_text_feature = ClinicalEncoder(5, 16)

        self.encoder_mri_t2w = MRIt2wEncoder(1, 8, 64)  # mono-modality
        self.encoder_mri_hbv = MRIhbvadcEncoder(1, 4, 32)  # mono-modality
        self.encoder_mri_adc = MRIhbvadcEncoder(1, 4, 32)  # mono-modality
        self.encoder_mri_hbv_adc = MRIhbvadcEncoder(2, 8, 64)  # cross-modality

        self.mlp = nn.Sequential(
            nn.Linear(64 + 64 + 64 + 16 + 64 + 32 + 32 + 64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1))

    def forward(self, x_mri_t2w, x_mri_hbv, x_mri_adc, x_clinical):  # , x_mri_hbv, x_mri_adc, x_clinical
        # x_mri_hbv = torch.ones((2, 1, 26, 128, 120), device=x_mri_t2w.device)
        # x_mri_adc = torch.ones((2, 1, 26, 128, 120), device=x_mri_t2w.device)
        # x_clinical = torch.ones((2, 18), device=x_mri_t2w.device)

        x_mri_hbv_adc = torch.concat((x_mri_hbv, x_mri_adc), dim=1)

        x_encoder_numeric_feature = self.encoder_numeric_feature(x_clinical[:, 0:3])
        x_encoder_ordinal_feature = self.encoder_ordinal_feature(x_clinical[:, 3:8])
        x_encoder_binary_feature = self.encoder_binary_feature(x_clinical[:, 8:13])
        x_encoder_text_feature = self.encoder_text_feature(x_clinical[:, 13:18])

        x_encoder_mri_t2w = self.encoder_mri_t2w(x_mri_t2w)
        x_encoder_mri_hbv = self.encoder_mri_hbv(x_mri_hbv)
        x_encoder_mri_adc = self.encoder_mri_adc(x_mri_adc)
        x_encoder_mri_hbv_adc = self.encoder_mri_hbv_adc(x_mri_hbv_adc)

        x_fusion = torch.concat((x_encoder_numeric_feature, x_encoder_ordinal_feature, x_encoder_binary_feature, x_encoder_text_feature,
                                 x_encoder_mri_t2w, x_encoder_mri_hbv, x_encoder_mri_adc, x_encoder_mri_hbv_adc), dim=1)

        risk = self.mlp(x_fusion)

        return risk.squeeze(1)



