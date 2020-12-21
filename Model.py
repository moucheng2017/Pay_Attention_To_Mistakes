import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================
# Networks
# =============================


class ERFANet(nn.Module):

    # Effective Receptive Attention Network

    def __init__(self, in_ch, width, class_no, attention_type, mode='all', identity_add=True):

        super(ERFANet, self).__init__()

        self.identity = identity_add

        if class_no == 2:
            self.final_in = 1
        else:
            self.final_in = class_no

        self.w1 = width
        self.w2 = width * 2
        self.w3 = width * 4
        self.w4 = width * 8

        if 'encoder' in mode or 'all' in mode:
            self.econv0 = single_conv(in_channels=in_ch, out_channels=self.w1, step=1)
            self.econv1 = MistakeAttention(in_channels=self.w1, out_channels=self.w2, step=2, addition=self.identity, attention_type=attention_type)
            self.econv2 = MistakeAttention(in_channels=self.w2, out_channels=self.w3, step=2, addition=self.identity, attention_type=attention_type)
            self.econv3 = MistakeAttention(in_channels=self.w3, out_channels=self.w4, step=2, addition=self.identity, attention_type=attention_type)
            self.bridge = MistakeAttention(in_channels=self.w4, out_channels=self.w4, step=1, addition=self.identity, attention_type=attention_type)
        else:
            self.econv0 = single_conv(in_channels=in_ch, out_channels=self.w1, step=1)
            self.econv1 = double_conv(in_channels=self.w1, out_channels=self.w2, step=2)
            self.econv2 = double_conv(in_channels=self.w2, out_channels=self.w3, step=2)
            self.econv3 = double_conv(in_channels=self.w3, out_channels=self.w4, step=2)
            self.bridge = double_conv(in_channels=self.w4, out_channels=self.w4, step=1)

        self.bridge_smooth = single_conv(in_channels=self.w4, out_channels=self.w4, step=1)

        if 'decoder' in mode or 'all' in mode:
            self.decoder3 = MistakeAttention(in_channels=self.w3, out_channels=self.w3, step=1, addition=self.identity, attention_type=attention_type)
            self.decoder2 = MistakeAttention(in_channels=self.w2, out_channels=self.w2, step=1, addition=self.identity, attention_type=attention_type)
            self.decoder1 = MistakeAttention(in_channels=self.w1, out_channels=self.w1, step=1, addition=self.identity, attention_type=attention_type)
            self.decoder0 = MistakeAttention(in_channels=self.w1, out_channels=self.w1, step=1, addition=self.identity, attention_type=attention_type)
        else:
            self.decoder3 = double_conv(in_channels=self.w3, out_channels=self.w3, step=1)
            self.decoder2 = double_conv(in_channels=self.w2, out_channels=self.w2, step=1)
            self.decoder1 = double_conv(in_channels=self.w1, out_channels=self.w1, step=1)
            self.decoder0 = double_conv(in_channels=self.w1, out_channels=self.w1, step=1)

        self.smooth3 = single_conv(in_channels=self.w4 + self.w4, out_channels=self.w3, step=1)
        self.smooth2 = single_conv(in_channels=self.w3 + self.w3, out_channels=self.w2, step=1)
        self.smooth1 = single_conv(in_channels=self.w2 + self.w2, out_channels=self.w1, step=1)
        self.smooth0 = single_conv(in_channels=self.w1 + self.w1, out_channels=self.w1, step=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_last = nn.Conv2d(self.w1, self.final_in, 1, bias=True)

    def forward(self, x):

        x0 = self.econv0(x)
        x1 = self.econv1(x0)
        x2 = self.econv2(x1)
        x3 = self.econv3(x2)
        x4 = self.bridge(x3)

        y = self.upsample(x4)
        y = self.bridge_smooth(y)

        if y.size()[2] != x3.size()[2]:

            diffY = torch.tensor([x3.size()[2] - y.size()[2]])
            diffX = torch.tensor([x3.size()[3] - y.size()[3]])
            #
            y = F.pad(y, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        y = torch.cat([y, x3], dim=1)
        y = self.upsample(y)
        y = self.smooth3(y)
        y = self.decoder3(y)

        y = torch.cat([y, x2], dim=1)
        y = self.upsample(y)
        y = self.smooth2(y)
        y = self.decoder2(y)

        y = torch.cat([y, x1], dim=1)
        y = self.upsample(y)
        y = self.smooth1(y)
        y = self.decoder1(y)

        y = torch.cat([y, x0], dim=1)
        y = self.smooth0(y)
        y = self.decoder0(y)

        y = self.conv_last(y)

        return y


# =============================
# Blocks
# =============================


def double_conv(in_channels, out_channels, step):

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.ReLU(inplace=True)
    )


def single_conv(in_channels, out_channels, step):

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.ReLU(inplace=True)
    )


class MistakeAttention(nn.Module):

    def __init__(self, in_channels, out_channels, step, addition, attention_type):
        super(MistakeAttention, self).__init__()
        self.addition = addition
        self.type = attention_type

        if 'FP' in self.type or 'fp' in self.type:

            dilation = 9

            self.main_branch = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=step, padding=1, bias=False),
                nn.InstanceNorm2d(num_features=out_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.InstanceNorm2d(num_features=out_channels, affine=True),
                nn.ReLU(inplace=True)
            )

            self.attention_branch = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=4 * out_channels, kernel_size=1, stride=step, dilation=1, padding=0, bias=False, groups=1),
                nn.Conv2d(in_channels=4 * out_channels, out_channels=4 * out_channels, kernel_size=3, stride=1, dilation=dilation, padding=dilation, bias=False, groups=4 * out_channels),
                nn.InstanceNorm2d(num_features=4 * out_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=4 * out_channels, out_channels=4 * out_channels, kernel_size=3, stride=1, dilation=dilation, padding=dilation, bias=False, groups=4 * out_channels),
                nn.InstanceNorm2d(num_features=4 * out_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=4 * out_channels, out_channels=out_channels, kernel_size=1, stride=1, dilation=1, padding=0, bias=False),
                nn.Sigmoid()
            )

        elif 'FN' in self.type or 'fn' in self.type:

            self.main_branch = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=step, padding=1, bias=False),
                nn.InstanceNorm2d(num_features=out_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.InstanceNorm2d(num_features=out_channels, affine=True),
                nn.ReLU(inplace=True)
            )

            self.attention_branch_1 = nn.Conv2d(in_channels=in_channels, out_channels=4 * out_channels, kernel_size=1, stride=step, dilation=1, padding=0, bias=False)

            self.attention_branch_identity = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=step, dilation=1, padding=0, bias=False)

            self.attention_branch_2 = nn.Sequential(
                nn.Conv2d(in_channels=4 * out_channels, out_channels=4 * out_channels, kernel_size=3, stride=1, dilation=1, padding=1, bias=False, groups=4 * out_channels),
                nn.InstanceNorm2d(num_features=4 * out_channels, affine=True),
                nn.ReLU(inplace=True)
            )

            self.attention_branch_3 = nn.Sequential(
                nn.Conv2d(in_channels=4 * out_channels, out_channels=4 * out_channels, kernel_size=3, stride=1, dilation=1, padding=1, bias=False, groups=4 * out_channels),
                nn.InstanceNorm2d(num_features=4 * out_channels, affine=True),
                nn.ReLU(inplace=True)
            )

            self.attention_branch_4 = nn.Conv2d(in_channels=4 * out_channels, out_channels=out_channels, kernel_size=1, stride=1, dilation=1, padding=0, bias=False)

            self.attention_branch_5 = nn.Sigmoid()

    def forward(self, x):

        if 'FP' in self.type or 'fp' in self.type:

            attention = self.attention_branch(x)
            features = self.main_branch(x)

        elif 'FN' in self.type or 'fn' in self.type:

            f1 = self.attention_branch_1(x)
            f2 = self.attention_branch_2(f1) + f1
            attention = self.attention_branch_3(f2) + f2
            attention = self.attention_branch_4(attention) + self.attention_branch_identity(x)
            attention = self.attention_branch_5(attention)
            features = self.main_branch(x)

        if self.addition is True:
            output = attention*features + features
        else:
            output = attention * features

        return output


# class FPA(nn.Module):
#
#     def __init__(self, in_channels, out_channels, step, addition):
#
#         super(FPA, self).__init__()
#
#         self.addition = addition
#
#         self.main_branch = nn.Sequential(
#             nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=step, padding=1, bias=False),
#             nn.InstanceNorm2d(num_features=out_channels, affine=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.InstanceNorm2d(num_features=out_channels, affine=True),
#             nn.ReLU(inplace=True)
#         )
#
#         self.attention_branch = nn.Sequential(
#             nn.Conv2d(in_channels=in_channels, out_channels=4*out_channels, kernel_size=1, stride=step, dilation=1, padding=0, bias=False, groups=1),
#             nn.Conv2d(in_channels=4*out_channels, out_channels=4*out_channels, kernel_size=3, stride=1, dilation=6, padding=6, bias=False, groups=4*out_channels),
#             nn.InstanceNorm2d(num_features=4*out_channels, affine=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=4*out_channels, out_channels=out_channels, kernel_size=1, stride=1, dilation=1, padding=0, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#
#         attention = self.attention_branch(x)
#         features = self.main_branch(x)
#
#         if self.addition is True:
#             output = attention*features + features
#         else:
#             output = attention * features
#
#         return output
#
#
# class FNA(nn.Module):
#
#     def __init__(self, in_channels, out_channels, step, addition):
#
#         super(FNA, self).__init__()
#
#         self.addition = addition
#
#         self.main_branch = nn.Sequential(
#             nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=step, padding=1, bias=False),
#             nn.InstanceNorm2d(num_features=out_channels, affine=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.InstanceNorm2d(num_features=out_channels, affine=True),
#             nn.ReLU(inplace=True)
#         )
#
#         self.attention_branch_1 = nn.Conv2d(in_channels=in_channels, out_channels=4*out_channels, kernel_size=1, stride=step, dilation=1, padding=0, bias=False)
#
#         self.attention_branch_identity = nn.Conv2d(in_channels=in_channels, out_channels=2*out_channels, kernel_size=1, stride=step, dilation=1, padding=0, bias=False)
#
#         self.attention_branch_2 = nn.Sequential(
#             nn.Conv2d(in_channels=4*out_channels, out_channels=4*out_channels, kernel_size=3, stride=1, dilation=1, padding=1, bias=False, groups=4*out_channels),
#             nn.InstanceNorm2d(num_features=4*out_channels, affine=True),
#             nn.ReLU(inplace=True)
#         )
#
#         self.attention_branch_3 = nn.Sequential(
#             nn.Conv2d(in_channels=4*out_channels, out_channels=4*out_channels, kernel_size=3, stride=1, dilation=1, padding=1, bias=False, groups=4*out_channels),
#             nn.InstanceNorm2d(num_features=4*out_channels, affine=True),
#             nn.ReLU(inplace=True)
#         )
#
#         self.attention_branch_4 = nn.Conv2d(in_channels=4*out_channels, out_channels=out_channels, kernel_size=1, stride=1, dilation=1, padding=0, bias=False)
#
#     def forward(self, x):
#
#         f1 = self.attention_branch_1(x)
#         f2 = self.attention_branch_2(f1) + f1
#         attention = self.attention_branch_3(f2) + f2
#         attention = self.attention_branch_4(attention) + self.attention_branch_identity(x)
#
#         features = self.main_branch(x)
#
#         if self.addition is True:
#             output = attention*features + features
#         else:
#             output = attention*features
#
#         return output
#
#
