import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============
# Basic modules:
# ==============


def first_conv(in_channels, out_channels, step):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 5, stride=step, padding=2, groups=1, bias=False),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.ReLU(inplace=True)
    )


def single_conv(in_channels, out_channels, step):
    #
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.ReLU(inplace=True)
    )


def double_conv(in_channels, out_channels, step):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=step,
                  padding=1, groups=1, bias=False),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, stride=1,
                  padding=1, groups=1, bias=False),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.ReLU(inplace=True)
    )


def dilated_conv(in_channels, out_channels, step, dilation):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=dilation, groups=1, dilation=dilation, bias=False),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=dilation, groups=1, dilation=dilation, bias=False),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.ReLU(inplace=True)
    )


# ==================
# Attention modules:
# ==================


class Attention_block(nn.Module):
    # Attention Unet
    # references:
    # Learn to Pay Attention, ICLR 2018
    # Attention U-Net: Learning Where to Look for the Pancreas, MIDL 2018
    def __init__(self, F_g, F_l):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_l // 2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(F_l // 2, affine=False)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_l // 2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(F_l // 2, affine=False)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_l // 2, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(1, affine=False),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return psi


class SE(nn.Module):
    # Squeeze-and-Excitation Networks, CVPR 2018
    def __init__(self, channel_no):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.squeeze = nn.Conv2d(channel_no, channel_no // 8, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.expand = nn.Conv2d(channel_no // 8, channel_no, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xx = self.avg_pool(x)
        channel_attention = self.sigmoid(self.expand(self.relu(self.squeeze(xx))))
        output = x*channel_attention + x
        return output


class CSE(nn.Module):
    # Spatial and channel squeeze
    # concurrent spatial and channel squeeze MICCAI 2018
    def __init__(self, channel_no):
        super(CSE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.squeeze = nn.Conv2d(channel_no, channel_no // 8, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.expand = nn.Conv2d(channel_no // 8, channel_no, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()
        #
        self.spatial_squeeze = nn.Conv2d(channel_no, 1, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #
        spatial_attention = self.sigmoid(self.spatial_squeeze(x)) * x
        #
        channel_attention = self.sigmoid(self.expand(self.relu(self.squeeze(self.avg_pool(x))))) * x
        output = spatial_attention + channel_attention
        #
        return output


class GE(nn.Module):
    # Gather Excite: exploiting feature context in convolutional neural network, NIPS 2018
    def __init__(self, channel_no):
        super(GE, self).__init__()
        self.squeeze_spatial_1 = nn.Conv2d(channel_no, channel_no, kernel_size=3, padding=1, stride=2, bias=False, groups=channel_no)
        self.squeeze_spatial_1_norm = nn.InstanceNorm2d(channel_no, affine=True)
        self.squeeze_spatial_2 = nn.Conv2d(channel_no, channel_no, kernel_size=3, padding=1, stride=2, bias=False, groups=channel_no)
        self.squeeze_spatial_2_norm = nn.InstanceNorm2d(channel_no, affine=True)
        # self.squeeze_spatial_3 = nn.Conv2d(channel_no, channel_no, kernel_size=3, padding=1, stride=2, bias=False, groups=channel_no)
        # self.squeeze_spatial_3_norm = nn.InstanceNorm2d(channel_no, affine=True)
        self.interpolate = nn.Upsample(scale_factor=4, mode='nearest')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.sigmoid(self.interpolate(self.squeeze_spatial_2_norm(self.squeeze_spatial_2(self.squeeze_spatial_1_norm(self.squeeze_spatial_1(x))))))
        output = x*attention + x
        return output


class CBAM(nn.Module):
    # Convolutional block attention module, ECCV 2018
    def __init__(self, channel_no):
        super(CBAM, self).__init__()
        self.avg_pool_channel = nn.AdaptiveAvgPool2d(1)
        self.max_pool_channel = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channel_no, channel_no // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(channel_no // 16, channel_no, 1, bias=False)
        self.sigmoid_c = nn.Sigmoid()
        #
        self.conv_spatial = nn.Conv2d(2, 1, 3, padding=1, bias=False)
        self.sigmoid_s = nn.Sigmoid()

    def forward(self, x):
        # channel attention:
        origin = x
        avg_c = self.fc2(self.relu1(self.fc1(self.avg_pool_channel(x))))
        max_c = self.fc2(self.relu1(self.fc1(self.max_pool_channel(x))))
        a_c = self.sigmoid_c((avg_c + max_c))
        x = x*a_c
        # spatial attention:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv_spatial(attention)
        attention = self.sigmoid_s(attention)
        output = attention*x + origin
        return output


class GCNonLocal(nn.Module):
    # GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond, arXiv
    def __init__(self, channel_no):
        super(GCNonLocal, self).__init__()
        self.conv_reduce = nn.Conv2d(channel_no, 1, kernel_size=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.squeeze = nn.Conv2d(channel_no, channel_no // 8, kernel_size=1, padding=0, bias=False)
        self.norm = nn.InstanceNorm2d(channel_no // 8, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.expand = nn.Conv2d(channel_no // 8, channel_no, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        xx = self.conv_reduce(x)
        xx = xx.view(b, h*w, 1)
        xx = self.softmax(xx)
        x_ = x.view(b, c, h*w)
        xxx = torch.bmm(x_, xx)
        xxx = xxx.view(b, c, 1, 1)
        attention = self.expand(self.relu(self.norm(self.squeeze(xxx))))
        output = attention*x + x
        return output


class DilatedUNet(nn.Module):
    # baseline 1:
    # u-net
    # dilation in encoder:
    def __init__(self, in_ch, width, dilation):
        super().__init__()
        class_no = 2
        #
        if class_no > 2:
            self.final_in = class_no
        else:
            self.final_in = 1
        #
        self.w1 = width
        self.w2 = width * 2
        self.w3 = width * 4
        self.w4 = width * 8
        #
        self.dconv_down1 = double_conv(in_ch, self.w1, step=1)
        self.dconv_down2 = dilated_conv(self.w1, self.w2, step=2, dilation=dilation)
        self.dconv_down3 = dilated_conv(self.w2, self.w3, step=2, dilation=dilation)
        self.dconv_down4 = dilated_conv(self.w3, self.w4, step=2, dilation=dilation)
        #
        self.bridge = double_conv(self.w4, self.w4, step=1)
        #
        self.dconv_up3 = double_conv(self.w3 + self.w4, self.w3, step=1)
        self.dconv_up2 = double_conv(self.w2 + self.w3, self.w2, step=1)
        self.dconv_up1 = double_conv(self.w1 + self.w2, self.w1, step=1)
        #
        self.max_pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #
        self.conv_last = nn.Conv2d(width, self.final_in, 1, bias=False)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        conv2 = self.dconv_down2(conv1)
        conv3 = self.dconv_down3(conv2)
        conv4 = self.dconv_down4(conv3)
        conv4 = self.bridge(conv4)
        x = self.upsample(conv4)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        out = self.conv_last(x)
        return out


# ============================


class CSE_UNet_Encoder(nn.Module):
    #
    def __init__(self, in_ch, width):
        super().__init__()
        class_no = 2
        if class_no > 2:
            self.final_in = class_no
        else:
            self.final_in = 1
        #
        self.w1 = width
        self.w2 = width * 2
        self.w3 = width * 4
        self.w4 = width * 8
        #
        self.dconv_up1 = double_conv(self.w1 + self.w2, self.w1, step=1)
        self.cse_1 = CSE(self.w1)
        self.dconv_up2 = double_conv(self.w2 + self.w3, self.w2, step=1)
        self.cse_2 = CSE(self.w2)
        self.dconv_up3 = double_conv(self.w3 + self.w4, self.w3, step=1)
        self.cse_3 = CSE(self.w3)
        self.bridge = double_conv(self.w4, self.w4, step=1)
        #
        self.dconv_down1 = double_conv(in_ch, self.w1, step=1)
        self.dconv_down2 = double_conv(self.w1, self.w2, step=2)
        self.dconv_down3 = double_conv(self.w2, self.w3, step=2)
        self.dconv_down4 = double_conv(self.w3, self.w4, step=2)
        #
        self.max_pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_last = nn.Conv2d(width, self.final_in, 1, bias=False)

    def forward(self, x):
        conv1 = self.cse_1(self.dconv_down1(x))
        conv2 = self.cse_2(self.dconv_down2(conv1))
        conv3 = self.cse_3(self.dconv_down3(conv2))
        conv4 = self.dconv_down4(conv3)
        conv4 = self.bridge(conv4)
        x = self.upsample(conv4)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        out = self.conv_last(x)
        return out


class CSE_UNet_Full(nn.Module):
    def __init__(self, in_ch, width):
        super().__init__()
        class_no = 2
        if class_no > 2:
            self.final_in = class_no
        else:
            self.final_in = 1
        self.w1 = width
        self.w2 = width * 2
        self.w3 = width * 4
        self.w4 = width * 8
        #
        self.dconv_up1 = double_conv(self.w1 + self.w2, self.w1, step=1)
        self.cse_1 = CSE(self.w1)
        self.cse_u1 = CSE(self.w1)
        self.dconv_up2 = double_conv(self.w2 + self.w3, self.w2, step=1)
        self.cse_2 = CSE(self.w2)
        self.cse_u2 = CSE(self.w2)
        self.dconv_up3 = double_conv(self.w3 + self.w4, self.w3, step=1)
        self.cse_3 = CSE(self.w3)
        self.cse_u3 = CSE(self.w3)
        self.bridge = double_conv(self.w4, self.w4, step=1)
        #
        self.dconv_down1 = double_conv(in_ch, self.w1, step=1)
        self.dconv_down2 = double_conv(self.w1, self.w2, step=2)
        self.dconv_down3 = double_conv(self.w2, self.w3, step=2)
        self.dconv_down4 = double_conv(self.w3, self.w4, step=2)
        #
        self.max_pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_last = nn.Conv2d(width, self.final_in, 1, bias=False)

    def forward(self, x):
        conv1 = self.cse_1(self.dconv_down1(x))
        conv2 = self.cse_2(self.dconv_down2(conv1))
        conv3 = self.cse_3(self.dconv_down3(conv2))
        conv4 = self.dconv_down4(conv3)
        conv4 = self.bridge(conv4)
        x = self.upsample(conv4)
        x = torch.cat([x, conv3], dim=1)
        x = self.cse_u3(self.dconv_up3(x))
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.cse_u2(self.dconv_up2(x))
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.cse_u1(self.dconv_up1(x))
        out = self.conv_last(x)
        return out


class Deeper_CSE_UNet_Full(nn.Module):
    def __init__(self, in_ch, width):
        super().__init__()
        class_no = 2
        if class_no > 2:
            self.final_in = class_no
        else:
            self.final_in = 1
        self.w1 = width
        self.w2 = width * 2
        self.w3 = width * 4
        self.w4 = width * 8
        #
        self.dconv_up0 = double_conv(self.w1 + self.w1, self.w1, step=1)
        self.dconv_up1 = double_conv(self.w1 + self.w2, self.w1, step=1)
        self.cse_1 = CSE(self.w1)
        self.cse_u1 = CSE(self.w1)
        self.dconv_up2 = double_conv(self.w2 + self.w3, self.w2, step=1)
        self.cse_2 = CSE(self.w2)
        self.cse_u2 = CSE(self.w2)
        self.dconv_up3 = double_conv(self.w3 + self.w4, self.w3, step=1)
        self.cse_3 = CSE(self.w3)
        self.cse_u3 = CSE(self.w3)
        self.bridge = double_conv(self.w4, self.w4, step=1)
        #
        self.dconv_down0 = first_conv(in_ch, self.w1, step=1)
        self.dconv_down1 = double_conv(self.w1, self.w1, step=2)
        self.dconv_down2 = double_conv(self.w1, self.w2, step=2)
        self.dconv_down3 = double_conv(self.w2, self.w3, step=2)
        self.dconv_down4 = double_conv(self.w3, self.w4, step=2)
        #
        self.max_pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_last = nn.Conv2d(width, self.final_in, 1, bias=False)

    def forward(self, x):
        conv0 = self.dconv_down0(x)
        conv1 = self.cse_1(self.dconv_down1(conv0))
        conv2 = self.cse_2(self.dconv_down2(conv1))
        conv3 = self.cse_3(self.dconv_down3(conv2))
        conv4 = self.dconv_down4(conv3)
        conv4 = self.bridge(conv4)
        x = self.upsample(conv4)
        x = torch.cat([x, conv3], dim=1)
        x = self.cse_u3(self.dconv_up3(x))
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.cse_u2(self.dconv_up2(x))
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.cse_u1(self.dconv_up1(x))
        x = self.upsample(x)
        x = torch.cat([x, conv0], dim=1)
        x = self.dconv_up0(x)
        out = self.conv_last(x)
        return out

# ==================================


class GCNonLocal_UNet_All(nn.Module):
    def __init__(self, in_ch, width):
        super().__init__()
        class_no = 2
        if class_no > 2:
            self.final_in = class_no
        else:
            self.final_in = 1
        self.w1 = width
        self.w2 = width * 2
        self.w3 = width * 4
        self.w4 = width * 8
        #
        self.dconv_up1 = double_conv(self.w1 + self.w2, self.w1, step=1)
        self.se_1 = GCNonLocal(self.w1)
        self.dconv_up2 = double_conv(self.w2 + self.w3, self.w2, step=1)
        self.se_2 = GCNonLocal(self.w2)
        self.dconv_up3 = double_conv(self.w3 + self.w4, self.w3, step=1)
        self.se_3 = GCNonLocal(self.w3)
        self.bridge = double_conv(self.w4, self.w4, step=1)
        self.se_4 = GCNonLocal(self.w4)
        #
        self.dconv_down1 = double_conv(in_ch, self.w1, step=1)
        self.se_1_d = GCNonLocal(self.w1)
        self.dconv_down2 = double_conv(self.w1, self.w2, step=2)
        self.se_2_d = GCNonLocal(self.w2)
        self.dconv_down3 = double_conv(self.w2, self.w3, step=2)
        self.se_3_d = GCNonLocal(self.w3)
        self.dconv_down4 = double_conv(self.w3, self.w4, step=2)
        #
        self.max_pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_last = nn.Conv2d(width, self.final_in, 1, bias=False)

    def forward(self, x):
        conv1 = self.se_1(self.dconv_down1(x))
        conv2 = self.se_2(self.dconv_down2(conv1))
        conv3 = self.se_3(self.dconv_down3(conv2))
        conv4 = self.dconv_down4(conv3)
        conv4 = self.se_4(self.bridge(conv4))
        x = self.upsample(conv4)
        x = torch.cat([x, conv3], dim=1)
        x = self.se_3_d(self.dconv_up3(x))
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.se_2_d(self.dconv_up2(x))
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.se_1_d(self.dconv_up1(x))
        out = self.conv_last(x)
        return out


class GCNonLocal_UNet_Decoder(nn.Module):
    def __init__(self, in_ch, width):
        super().__init__()
        class_no = 2
        if class_no > 2:
            self.final_in = class_no
        else:
            self.final_in = 1
        self.w1 = width
        self.w2 = width * 2
        self.w3 = width * 4
        self.w4 = width * 8
        #
        self.dconv_up1 = double_conv(self.w1 + self.w2, self.w1, step=1)
        self.se_1 = GCNonLocal(self.w1)
        self.dconv_up2 = double_conv(self.w2 + self.w3, self.w2, step=1)
        self.se_2 = GCNonLocal(self.w2)
        self.dconv_up3 = double_conv(self.w3 + self.w4, self.w3, step=1)
        self.se_3 = GCNonLocal(self.w3)
        self.bridge = double_conv(self.w4, self.w4, step=1)
        self.dconv_down1 = double_conv(in_ch, self.w1, step=1)
        self.dconv_down2 = double_conv(self.w1, self.w2, step=2)
        self.dconv_down3 = double_conv(self.w2, self.w3, step=2)
        self.dconv_down4 = double_conv(self.w3, self.w4, step=2)
        #
        self.max_pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_last = nn.Conv2d(width, self.final_in, 1, bias=False)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        conv2 = self.dconv_down2(conv1)
        conv3 = self.dconv_down3(conv2)
        conv4 = self.dconv_down4(conv3)
        conv4 = self.bridge(conv4)
        x = self.upsample(conv4)
        x = torch.cat([x, conv3], dim=1)
        x = self.se_3(self.dconv_up3(x))
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.se_2(self.dconv_up2(x))
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.se_1(self.dconv_up1(x))
        out = self.conv_last(x)
        return out


class GCNonLocal_UNet_Encoder(nn.Module):
    def __init__(self, in_ch, width):
        super().__init__()
        class_no = 2
        if class_no > 2:
            self.final_in = class_no
        else:
            self.final_in = 1
        self.w1 = width
        self.w2 = width * 2
        self.w3 = width * 4
        self.w4 = width * 8
        #
        self.dconv_up1 = double_conv(self.w1 + self.w2, self.w1, step=1)
        self.se_1 = GCNonLocal(self.w1)
        self.dconv_up2 = double_conv(self.w2 + self.w3, self.w2, step=1)
        self.se_2 = GCNonLocal(self.w2)
        self.dconv_up3 = double_conv(self.w3 + self.w4, self.w3, step=1)
        self.se_3 = GCNonLocal(self.w3)
        self.bridge = double_conv(self.w4, self.w4, step=1)
        self.se_4 = GCNonLocal(self.w4)
        self.dconv_down1 = double_conv(in_ch, self.w1, step=1)
        self.dconv_down2 = double_conv(self.w1, self.w2, step=2)
        self.dconv_down3 = double_conv(self.w2, self.w3, step=2)
        self.dconv_down4 = double_conv(self.w3, self.w4, step=2)
        #
        self.max_pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_last = nn.Conv2d(width, self.final_in, 1, bias=False)

    def forward(self, x):
        conv1 = self.se_1(self.dconv_down1(x))
        conv2 = self.se_2(self.dconv_down2(conv1))
        conv3 = self.se_3(self.dconv_down3(conv2))
        conv4 = self.dconv_down4(conv3)
        conv4 = self.se_4(self.bridge(conv4))
        x = self.upsample(conv4)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        out = self.conv_last(x)
        return out


# ============================


class CBAM_UNet_All(nn.Module):
    def __init__(self, in_ch, width):
        super().__init__()
        class_no = 2
        if class_no > 2:
            self.final_in = class_no
        else:
            self.final_in = 1
        self.w1 = width
        self.w2 = width * 2
        self.w3 = width * 4
        self.w4 = width * 8
        #
        self.dconv_up1 = double_conv(self.w1 + self.w2, self.w1, step=1)
        self.se_1 = CBAM(self.w1)
        self.dconv_up2 = double_conv(self.w2 + self.w3, self.w2, step=1)
        self.se_2 = CBAM(self.w2)
        self.dconv_up3 = double_conv(self.w3 + self.w4, self.w3, step=1)
        self.se_3 = CBAM(self.w3)
        self.bridge = double_conv(self.w4, self.w4, step=1)
        self.se_4 = CBAM(self.w4)
        #
        self.dconv_down1 = double_conv(in_ch, self.w1, step=1)
        self.se_1_d = CBAM(self.w1)
        self.dconv_down2 = double_conv(self.w1, self.w2, step=2)
        self.se_2_d = CBAM(self.w2)
        self.dconv_down3 = double_conv(self.w2, self.w3, step=2)
        self.se_3_d = CBAM(self.w3)
        self.dconv_down4 = double_conv(self.w3, self.w4, step=2)
        #
        self.max_pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_last = nn.Conv2d(width, self.final_in, 1, bias=False)

    def forward(self, x):
        conv1 = self.se_1(self.dconv_down1(x))
        conv2 = self.se_2(self.dconv_down2(conv1))
        conv3 = self.se_3(self.dconv_down3(conv2))
        conv4 = self.dconv_down4(conv3)
        conv4 = self.se_4(self.bridge(conv4))
        x = self.upsample(conv4)
        x = torch.cat([x, conv3], dim=1)
        x = self.se_3_d(self.dconv_up3(x))
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.se_2_d(self.dconv_up2(x))
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.se_1_d(self.dconv_up1(x))
        out = self.conv_last(x)
        return out


class Deeper_CBAM_UNet_All(nn.Module):
    def __init__(self, in_ch, width):
        super().__init__()
        class_no = 2
        if class_no > 2:
            self.final_in = class_no
        else:
            self.final_in = 1
        self.w1 = width
        self.w2 = width * 2
        self.w3 = width * 4
        self.w4 = width * 8
        #
        self.dconv_up0 = first_conv(self.w1 + self.w1, self.w1, step=1)
        self.dconv_up1 = double_conv(self.w1 + self.w2, self.w1, step=1)
        self.se_1 = CBAM(self.w1)
        self.dconv_up2 = double_conv(self.w2 + self.w3, self.w2, step=1)
        self.se_2 = CBAM(self.w2)
        self.dconv_up3 = double_conv(self.w3 + self.w4, self.w3, step=1)
        self.se_3 = CBAM(self.w3)
        self.bridge = double_conv(self.w4, self.w4, step=1)
        self.se_4 = CBAM(self.w4)
        #
        self.dconv_down0 = first_conv(in_ch, self.w1, step=1)
        self.dconv_down1 = double_conv(self.w1, self.w1, step=2)
        self.se_1_d = CBAM(self.w1)
        self.dconv_down2 = double_conv(self.w1, self.w2, step=2)
        self.se_2_d = CBAM(self.w2)
        self.dconv_down3 = double_conv(self.w2, self.w3, step=2)
        self.se_3_d = CBAM(self.w3)
        self.dconv_down4 = double_conv(self.w3, self.w4, step=2)
        #
        self.max_pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_last = nn.Conv2d(width, self.final_in, 1, bias=False)

    def forward(self, x):
        conv0 = self.dconv_down0(x)
        conv1 = self.se_1(self.dconv_down1(conv0))
        conv2 = self.se_2(self.dconv_down2(conv1))
        conv3 = self.se_3(self.dconv_down3(conv2))
        conv4 = self.dconv_down4(conv3)
        conv4 = self.se_4(self.bridge(conv4))
        x = self.upsample(conv4)
        x = torch.cat([x, conv3], dim=1)
        x = self.se_3_d(self.dconv_up3(x))
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.se_2_d(self.dconv_up2(x))
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.se_1_d(self.dconv_up1(x))
        x = self.upsample(x)
        x = torch.cat([x, conv0], dim=1)
        x = self.dconv_up0(x)
        out = self.conv_last(x)
        return out


class CBAM_UNet_Decoder(nn.Module):
    def __init__(self, in_ch, width):
        super().__init__()
        class_no = 2
        if class_no > 2:
            self.final_in = class_no
        else:
            self.final_in = 1
        self.w1 = width
        self.w2 = width * 2
        self.w3 = width * 4
        self.w4 = width * 8
        #
        self.dconv_up1 = double_conv(self.w1 + self.w2, self.w1, step=1)
        self.se_1 = CBAM(self.w1)
        self.dconv_up2 = double_conv(self.w2 + self.w3, self.w2, step=1)
        self.se_2 = CBAM(self.w2)
        self.dconv_up3 = double_conv(self.w3 + self.w4, self.w3, step=1)
        self.se_3 = CBAM(self.w3)
        self.bridge = double_conv(self.w4, self.w4, step=1)
        self.dconv_down1 = double_conv(in_ch, self.w1, step=1)
        self.dconv_down2 = double_conv(self.w1, self.w2, step=2)
        self.dconv_down3 = double_conv(self.w2, self.w3, step=2)
        self.dconv_down4 = double_conv(self.w3, self.w4, step=2)
        #
        self.max_pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_last = nn.Conv2d(width, self.final_in, 1, bias=False)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        conv2 = self.dconv_down2(conv1)
        conv3 = self.dconv_down3(conv2)
        conv4 = self.dconv_down4(conv3)
        conv4 = self.bridge(conv4)
        x = self.upsample(conv4)
        x = torch.cat([x, conv3], dim=1)
        x = self.se_3(self.dconv_up3(x))
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.se_2(self.dconv_up2(x))
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.se_1(self.dconv_up1(x))
        out = self.conv_last(x)
        return out


class CBAM_UNet_Encoder(nn.Module):
    def __init__(self, in_ch, width):
        super().__init__()
        class_no = 2
        if class_no > 2:
            self.final_in = class_no
        else:
            self.final_in = 1
        self.w1 = width
        self.w2 = width * 2
        self.w3 = width * 4
        self.w4 = width * 8
        #
        self.dconv_up1 = double_conv(self.w1 + self.w2, self.w1, step=1)
        self.se_1 = CBAM(self.w1)
        self.dconv_up2 = double_conv(self.w2 + self.w3, self.w2, step=1)
        self.se_2 = CBAM(self.w2)
        self.dconv_up3 = double_conv(self.w3 + self.w4, self.w3, step=1)
        self.se_3 = CBAM(self.w3)
        self.bridge = double_conv(self.w4, self.w4, step=1)
        self.se_4 = CBAM(self.w4)
        self.dconv_down1 = double_conv(in_ch, self.w1, step=1)
        self.dconv_down2 = double_conv(self.w1, self.w2, step=2)
        self.dconv_down3 = double_conv(self.w2, self.w3, step=2)
        self.dconv_down4 = double_conv(self.w3, self.w4, step=2)
        #
        self.max_pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_last = nn.Conv2d(width, self.final_in, 1, bias=False)

    def forward(self, x):
        conv1 = self.se_1(self.dconv_down1(x))
        conv2 = self.se_2(self.dconv_down2(conv1))
        conv3 = self.se_3(self.dconv_down3(conv2))
        conv4 = self.dconv_down4(conv3)
        conv4 = self.se_4(self.bridge(conv4))
        x = self.upsample(conv4)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        out = self.conv_last(x)
        return out


# ===========================


class GE_UNet_All(nn.Module):
    def __init__(self, in_ch, width):
        super().__init__()
        class_no = 2
        if class_no > 2:
            self.final_in = class_no
        else:
            self.final_in = 1
        self.w1 = width
        self.w2 = width * 2
        self.w3 = width * 4
        self.w4 = width * 8
        #
        self.dconv_up1 = double_conv(self.w1 + self.w2, self.w1, step=1)
        self.se_1 = GE(self.w1)
        self.dconv_up2 = double_conv(self.w2 + self.w3, self.w2, step=1)
        self.se_2 = GE(self.w2)
        self.dconv_up3 = double_conv(self.w3 + self.w4, self.w3, step=1)
        self.se_3 = GE(self.w3)
        self.bridge = double_conv(self.w4, self.w4, step=1)
        self.se_4 = GE(self.w4)
        #
        self.dconv_down1 = double_conv(in_ch, self.w1, step=1)
        self.se_1_d = GE(self.w1)
        self.dconv_down2 = double_conv(self.w1, self.w2, step=2)
        self.se_2_d = GE(self.w2)
        self.dconv_down3 = double_conv(self.w2, self.w3, step=2)
        self.se_3_d = GE(self.w3)
        self.dconv_down4 = double_conv(self.w3, self.w4, step=2)
        #
        self.max_pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_last = nn.Conv2d(width, self.final_in, 1, bias=False)

    def forward(self, x):
        conv1 = self.se_1(self.dconv_down1(x))
        conv2 = self.se_2(self.dconv_down2(conv1))
        conv3 = self.se_3(self.dconv_down3(conv2))
        conv4 = self.dconv_down4(conv3)
        conv4 = self.se_4(self.bridge(conv4))
        x = self.upsample(conv4)
        x = torch.cat([x, conv3], dim=1)
        x = self.se_3_d(self.dconv_up3(x))
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.se_2_d(self.dconv_up2(x))
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.se_1_d(self.dconv_up1(x))
        out = self.conv_last(x)
        return out


class GE_UNet_Decoder(nn.Module):
    def __init__(self, in_ch, width):
        super().__init__()
        class_no = 2
        if class_no > 2:
            self.final_in = class_no
        else:
            self.final_in = 1
        self.w1 = width
        self.w2 = width * 2
        self.w3 = width * 4
        self.w4 = width * 8
        #
        self.dconv_up1 = double_conv(self.w1 + self.w2, self.w1, step=1)
        self.se_1 = GE(self.w1)
        self.dconv_up2 = double_conv(self.w2 + self.w3, self.w2, step=1)
        self.se_2 = GE(self.w2)
        self.dconv_up3 = double_conv(self.w3 + self.w4, self.w3, step=1)
        self.se_3 = GE(self.w3)
        self.bridge = double_conv(self.w4, self.w4, step=1)
        self.dconv_down1 = double_conv(in_ch, self.w1, step=1)
        self.dconv_down2 = double_conv(self.w1, self.w2, step=2)
        self.dconv_down3 = double_conv(self.w2, self.w3, step=2)
        self.dconv_down4 = double_conv(self.w3, self.w4, step=2)
        #
        self.max_pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_last = nn.Conv2d(width, self.final_in, 1, bias=False)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        conv2 = self.dconv_down2(conv1)
        conv3 = self.dconv_down3(conv2)
        conv4 = self.dconv_down4(conv3)
        conv4 = self.bridge(conv4)
        x = self.upsample(conv4)
        x = torch.cat([x, conv3], dim=1)
        x = self.se_3(self.dconv_up3(x))
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.se_2(self.dconv_up2(x))
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.se_1(self.dconv_up1(x))
        out = self.conv_last(x)
        return out


class GE_UNet_Encoder(nn.Module):
    def __init__(self, in_ch, width):
        super().__init__()
        class_no = 2
        if class_no > 2:
            self.final_in = class_no
        else:
            self.final_in = 1
        self.w1 = width
        self.w2 = width * 2
        self.w3 = width * 4
        self.w4 = width * 8
        #
        self.dconv_up1 = double_conv(self.w1 + self.w2, self.w1, step=1)
        self.se_1 = GE(self.w1)
        self.dconv_up2 = double_conv(self.w2 + self.w3, self.w2, step=1)
        self.se_2 = GE(self.w2)
        self.dconv_up3 = double_conv(self.w3 + self.w4, self.w3, step=1)
        self.se_3 = GE(self.w3)
        self.bridge = double_conv(self.w4, self.w4, step=1)
        self.se_4 = GE(self.w4)
        self.dconv_down1 = double_conv(in_ch, self.w1, step=1)
        self.dconv_down2 = double_conv(self.w1, self.w2, step=2)
        self.dconv_down3 = double_conv(self.w2, self.w3, step=2)
        self.dconv_down4 = double_conv(self.w3, self.w4, step=2)
        #
        self.max_pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_last = nn.Conv2d(width, self.final_in, 1, bias=False)

    def forward(self, x):
        conv1 = self.se_1(self.dconv_down1(x))
        conv2 = self.se_2(self.dconv_down2(conv1))
        conv3 = self.se_3(self.dconv_down3(conv2))
        conv4 = self.dconv_down4(conv3)
        conv4 = self.se_4(self.bridge(conv4))
        x = self.upsample(conv4)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        out = self.conv_last(x)
        return out

# ============================


class SE_UNet_All(nn.Module):
    def __init__(self, in_ch, width):
        super().__init__()
        class_no = 2
        if class_no > 2:
            self.final_in = class_no
        else:
            self.final_in = 1
        self.w1 = width
        self.w2 = width * 2
        self.w3 = width * 4
        self.w4 = width * 8
        #
        self.dconv_up1 = double_conv(self.w1 + self.w2, self.w1, step=1)
        self.se_1 = SE(self.w1)
        self.dconv_up2 = double_conv(self.w2 + self.w3, self.w2, step=1)
        self.se_2 = SE(self.w2)
        self.dconv_up3 = double_conv(self.w3 + self.w4, self.w3, step=1)
        self.se_3 = SE(self.w3)
        self.bridge = double_conv(self.w4, self.w4, step=1)
        self.se_4 = SE(self.w4)
        #
        self.dconv_down1 = double_conv(in_ch, self.w1, step=1)
        self.se_1_d = SE(self.w1)
        self.dconv_down2 = double_conv(self.w1, self.w2, step=2)
        self.se_2_d = SE(self.w2)
        self.dconv_down3 = double_conv(self.w2, self.w3, step=2)
        self.se_3_d = SE(self.w3)
        self.dconv_down4 = double_conv(self.w3, self.w4, step=2)
        #
        self.max_pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_last = nn.Conv2d(width, self.final_in, 1, bias=False)

    def forward(self, x):
        conv1 = self.se_1(self.dconv_down1(x))
        conv2 = self.se_2(self.dconv_down2(conv1))
        conv3 = self.se_3(self.dconv_down3(conv2))
        conv4 = self.dconv_down4(conv3)
        conv4 = self.se_4(self.bridge(conv4))
        x = self.upsample(conv4)
        x = torch.cat([x, conv3], dim=1)
        x = self.se_3_d(self.dconv_up3(x))
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.se_2_d(self.dconv_up2(x))
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.se_1_d(self.dconv_up1(x))
        out = self.conv_last(x)
        return out


class SE_UNet_Decoder(nn.Module):
    def __init__(self, in_ch, width):
        super().__init__()
        class_no = 2
        if class_no > 2:
            self.final_in = class_no
        else:
            self.final_in = 1
        self.w1 = width
        self.w2 = width * 2
        self.w3 = width * 4
        self.w4 = width * 8
        #
        self.dconv_up1 = double_conv(self.w1 + self.w2, self.w1, step=1)
        self.se_1 = SE(self.w1)
        self.dconv_up2 = double_conv(self.w2 + self.w3, self.w2, step=1)
        self.se_2 = SE(self.w2)
        self.dconv_up3 = double_conv(self.w3 + self.w4, self.w3, step=1)
        self.se_3 = SE(self.w3)
        self.bridge = double_conv(self.w4, self.w4, step=1)
        self.dconv_down1 = double_conv(in_ch, self.w1, step=1)
        self.dconv_down2 = double_conv(self.w1, self.w2, step=2)
        self.dconv_down3 = double_conv(self.w2, self.w3, step=2)
        self.dconv_down4 = double_conv(self.w3, self.w4, step=2)
        #
        self.max_pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_last = nn.Conv2d(width, self.final_in, 1, bias=False)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        conv2 = self.dconv_down2(conv1)
        conv3 = self.dconv_down3(conv2)
        conv4 = self.dconv_down4(conv3)
        conv4 = self.bridge(conv4)
        x = self.upsample(conv4)
        x = torch.cat([x, conv3], dim=1)
        x = self.se_3(self.dconv_up3(x))
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.se_2(self.dconv_up2(x))
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.se_1(self.dconv_up1(x))
        out = self.conv_last(x)
        return out


class SE_UNet_Encoder(nn.Module):
    def __init__(self, in_ch, width):
        super().__init__()
        class_no = 2
        if class_no > 2:
            self.final_in = class_no
        else:
            self.final_in = 1
        self.w1 = width
        self.w2 = width * 2
        self.w3 = width * 4
        self.w4 = width * 8
        #
        self.dconv_up1 = double_conv(self.w1 + self.w2, self.w1, step=1)
        self.se_1 = SE(self.w1)
        self.dconv_up2 = double_conv(self.w2 + self.w3, self.w2, step=1)
        self.se_2 = SE(self.w2)
        self.dconv_up3 = double_conv(self.w3 + self.w4, self.w3, step=1)
        self.se_3 = SE(self.w3)
        self.bridge = double_conv(self.w4, self.w4, step=1)
        self.se_4 = SE(self.w4)
        self.dconv_down1 = double_conv(in_ch, self.w1, step=1)
        self.dconv_down2 = double_conv(self.w1, self.w2, step=2)
        self.dconv_down3 = double_conv(self.w2, self.w3, step=2)
        self.dconv_down4 = double_conv(self.w3, self.w4, step=2)
        #
        self.max_pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_last = nn.Conv2d(width, self.final_in, 1, bias=False)

    def forward(self, x):
        conv1 = self.se_1(self.dconv_down1(x))
        conv2 = self.se_2(self.dconv_down2(conv1))
        conv3 = self.se_3(self.dconv_down3(conv2))
        conv4 = self.dconv_down4(conv3)
        conv4 = self.se_4(self.bridge(conv4))
        x = self.upsample(conv4)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        out = self.conv_last(x)
        return out


class UNet(nn.Module):
    #
    def __init__(self, in_ch, width, class_no):
        #
        super(UNet, self).__init__()
        #
        if class_no == 2:
            #
            self.final_in = 1
            #
        else:
            #
            self.final_in = class_no
        #
        self.w1 = width
        self.w2 = width * 2
        self.w3 = width * 4
        self.w4 = width * 8
        #
        self.econv0 = single_conv(in_channels=in_ch, out_channels=self.w1, step=1)
        self.econv1 = double_conv(in_channels=self.w1, out_channels=self.w2, step=2)
        self.econv2 = double_conv(in_channels=self.w2, out_channels=self.w3, step=2)
        self.econv3 = double_conv(in_channels=self.w3, out_channels=self.w4, step=2)
        self.bridge = double_conv(in_channels=self.w4, out_channels=self.w4, step=1)
        #
        self.dconv3 = double_conv(in_channels=self.w4+self.w4, out_channels=self.w3, step=1)
        self.dconv2 = double_conv(in_channels=self.w3+self.w3, out_channels=self.w2, step=1)
        self.dconv1 = double_conv(in_channels=self.w2+self.w2, out_channels=self.w1, step=1)
        self.dconv0 = double_conv(in_channels=self.w1+self.w1, out_channels=self.w1, step=1)
        #
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dconv_last = nn.Conv2d(self.w1, self.final_in, 1, bias=True)

    def forward(self, x):

        x0 = self.econv0(x)
        x1 = self.econv1(x0)
        x2 = self.econv2(x1)
        x3 = self.econv3(x2)
        x4 = self.bridge(x3)

        y = self.upsample(x4)

        if y.size()[2] != x3.size()[2]:

            diffY = torch.tensor([x3.size()[2] - y.size()[2]])
            diffX = torch.tensor([x3.size()[3] - y.size()[3]])
            #
            y = F.pad(y, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        y3 = torch.cat([y, x3], dim=1)
        y3 = self.dconv3(y3)
        y2 = self.upsample(y3)
        y2 = torch.cat([y2, x2], dim=1)
        y2 = self.dconv2(y2)
        y1 = self.upsample(y2)
        y1 = torch.cat([y1, x1], dim=1)
        y1 = self.dconv1(y1)
        y0 = self.upsample(y1)
        y0 = torch.cat([y0, x0], dim=1)
        y0 = self.dconv0(y0)
        y = self.dconv_last(y0)
        return y


class DeeperUNet(nn.Module):
    # baseline 1 for wmh:
    # u-net
    # improvements over original one:
    # instance normalisation
    # interpolation upsampling rather than de-conv
    # 4 stages
    #
    def __init__(self, in_ch, width):
        super().__init__()
        class_no = 2
        if class_no > 2:
            self.final_in = class_no
        else:
            self.final_in = 1
        self.w1 = width
        self.w2 = width * 2
        self.w3 = width * 4
        self.w4 = width * 8
        #
        self.dconv_down0 = first_conv(in_ch, self.w1, step=1)
        self.dconv_down1 = double_conv(self.w1, self.w1, step=2)
        self.dconv_down2 = double_conv(self.w1, self.w2, step=2)
        self.dconv_down3 = double_conv(self.w2, self.w3, step=2)
        self.dconv_down4 = double_conv(self.w3, self.w4, step=2)
        #
        self.bridge = double_conv(self.w4, self.w4, step=1)
        #
        self.dconv_up3 = double_conv(self.w3 + self.w4, self.w3, step=1)
        self.dconv_up2 = double_conv(self.w2 + self.w3, self.w2, step=1)
        self.dconv_up1 = double_conv(self.w1 + self.w2, self.w1, step=1)
        self.dconv_up0 = double_conv(self.w1 + self.w1, self.w1, step=1)
        #
        self.max_pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #
        self.conv_last = nn.Conv2d(width, self.final_in, 1, bias=False)

    def forward(self, x):
        conv0 = self.dconv_down0(x)
        conv1 = self.dconv_down1(conv0)
        conv2 = self.dconv_down2(conv1)
        conv3 = self.dconv_down3(conv2)
        conv4 = self.dconv_down4(conv3)
        conv4 = self.bridge(conv4)
        x = self.upsample(conv4)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        x = self.upsample(x)
        x = torch.cat([x, conv0], dim=1)
        x = self.dconv_up0(x)
        out = self.conv_last(x)
        return out


class AttentionUNet(nn.Module):
    def __init__(self, in_ch, width):
        super(AttentionUNet, self).__init__()
        # self.attention_visual = visulisation
        class_no = 2
        if class_no > 2:
            self.final_in = class_no
        else:
            self.final_in = 1

        self.w1 = width
        self.w2 = width * 2
        self.w3 = width * 4
        self.w4 = width * 8

        self.dconv_down1 = double_conv(in_ch, self.w1, step=1)
        self.dconv_down2 = double_conv(self.w1, self.w2, step=2)
        self.dconv_down3 = double_conv(self.w2, self.w3, step=2)
        self.dconv_down4 = double_conv(self.w3, self.w4, step=2)

        self.bridge = double_conv(self.w4, self.w4, step=1)
        self.dconv_up3 = double_conv(self.w3 + self.w4, self.w3, step=1)
        self.dconv_up2 = double_conv(self.w2 + self.w3, self.w2, step=1)
        self.dconv_up1 = double_conv(self.w1 + self.w2, self.w1, step=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.a3_match_res = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.a2_match_res = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.a1_match_res = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.attention_3 = Attention_block(self.w4, self.w3)

        self.attention_2 = Attention_block(self.w3, self.w2)

        self.attention_1 = Attention_block(self.w2, self.w1)

        self.conv_last = nn.Conv2d(width, self.final_in, 1, bias=False)

    def forward(self, x):
        #
        # if self.attention_visual is True:
        #     attention_weights = []
        #     trunk_features = []
        #
        s1 = self.dconv_down1(x)
        s2 = self.dconv_down2(s1)
        s3 = self.dconv_down3(s2)
        s4 = self.dconv_down4(s3)
        s4 = self.bridge(s4)
        #
        attn_3 = self.attention_3(self.a3_match_res(s4), s3)
        a_s3 = attn_3 * s3 + s3
        #
        # if self.attention_visual is True:
        #     attention_weights.append(attn_3)
        #     trunk_features.append(s3)
        #
        output = torch.cat([a_s3, self.upsample(s4)], dim=1)
        output = self.dconv_up3(output)
        #
        attn_2 = self.attention_2(self.a2_match_res(output), s2)
        a_s2 = attn_2 * s2 + s2
        #
        # if self.attention_visual is True:
        #     attention_weights.append(attn_2)
        #     trunk_features.append(s2)
        #
        output = torch.cat([a_s2, self.upsample(output)], dim=1)
        output = self.dconv_up2(output)
        #
        attn_1 = self.attention_1(self.a1_match_res(output), s1)
        a_s1 = attn_1 * s1 + s1
        #
        # if self.attention_visual is True:
        #     attention_weights.append(attn_1)
        #     trunk_features.append(s1)
        #
        output = torch.cat([a_s1, self.upsample(output)], dim=1)
        output = self.dconv_up1(output)
        output = self.conv_last(output)
        #
        return output
        # if self.attention_visual is False:
        #     return output
        # else:
        #     return output, attention_weights, trunk_features




