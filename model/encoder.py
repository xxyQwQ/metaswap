import torch
from torch import nn


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, input):
        output = self.network(input)
        return output


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.network = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, input, injection=None):
        output = self.network(input)
        if injection is not None:
            output = torch.cat((output, injection), dim=1)
        return output


class AttriduteEncoder(nn.Module):
    def __init__(self):
        super(AttriduteEncoder, self).__init__()

        self.downsample_1 = DownsampleBlock(3, 32)
        self.downsample_2 = DownsampleBlock(32, 64)
        self.downsample_3 = DownsampleBlock(64, 128)
        self.downsample_4 = DownsampleBlock(128, 256)
        self.downsample_5 = DownsampleBlock(256, 512)
        self.downsample_6 = DownsampleBlock(512, 1024)
        self.downsample_7 = DownsampleBlock(1024, 1024)

        self.upsample_1 = UpsampleBlock(1024, 1024)
        self.upsample_2 = UpsampleBlock(2048, 512)
        self.upsample_3 = UpsampleBlock(1024, 256)
        self.upsample_4 = UpsampleBlock(512, 128)
        self.upsample_5 = UpsampleBlock(256, 64)
        self.upsample_6 = UpsampleBlock(128, 32)
        self.upsample_7 = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, input):
        feature_1 = self.downsample_1(input)                    # (batch_size, 32, 128, 128)
        feature_2 = self.downsample_2(feature_1)                # (batch_size, 64, 64, 64)
        feature_3 = self.downsample_3(feature_2)                # (batch_size, 128, 32, 32)
        feature_4 = self.downsample_4(feature_3)                # (batch_size, 256, 16, 16)
        feature_5 = self.downsample_5(feature_4)                # (batch_size, 512, 8, 8)
        feature_6 = self.downsample_6(feature_5)                # (batch_size, 1024, 4, 4)
        attribute_1 = self.downsample_7(feature_6)              # (batch_size, 1024, 2, 2)
        
        attribute_2 = self.upsample_1(attribute_1, feature_6)   # (batch_size, 2048, 4, 4)
        attribute_3 = self.upsample_2(attribute_2, feature_5)   # (batch_size, 1024, 8, 8)
        attribute_4 = self.upsample_3(attribute_3, feature_4)   # (batch_size, 512, 16, 16)
        attribute_5 = self.upsample_4(attribute_4, feature_3)   # (batch_size, 256, 32, 32)
        attribute_6 = self.upsample_5(attribute_5, feature_2)   # (batch_size, 128, 64, 64)
        attribute_7 = self.upsample_6(attribute_6, feature_1)   # (batch_size, 64, 128, 128)
        attribute_8 = self.upsample_7(attribute_7)              # (batch_size, 64, 256, 256)

        return attribute_1, attribute_2, attribute_3, attribute_4, attribute_5, attribute_6, attribute_7, attribute_8
