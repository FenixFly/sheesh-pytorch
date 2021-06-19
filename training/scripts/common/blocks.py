import torch
import torch.nn as nn
import torch.nn.functional as F




class Residual_bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample = None, kernel_size = None, padding = None):
        super(Residual_bottleneck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.downsample = downsample

        compression = 4

        if kernel_size is None:
            kernel_size = 3
            padding = 1

        self.conv_pre = nn.Conv1d(in_channels=in_channels, out_channels=out_channels//compression, kernel_size=1, stride=stride,
                               padding=0, bias=False)
        self.conv1 = nn.Conv1d(in_channels=out_channels//compression, out_channels=out_channels//compression, kernel_size=kernel_size, stride=1, padding=padding, bias=False, groups=1)

        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.norm1 = nn.BatchNorm1d(num_features=out_channels//compression,affine=True)
        self.norm2 = nn.BatchNorm1d(num_features=out_channels//compression,affine=True)
        self.conv_post = nn.Conv1d(in_channels=out_channels // compression, out_channels=out_channels,
                               kernel_size=1, padding=0, bias=False)
        self.norm_post = nn.BatchNorm1d(num_features=out_channels,affine=True)

    def forward(self, x):

        if self.downsample is not None:
            x = self.downsample(x)

        out = x
        out = self.conv_pre(out)
        out = self.norm1(out)
        out = self.relu1(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu2(out)

        out = self.conv_post(out)
        out = self.norm_post(out)

        out = x + out

        return out


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample = None, kernel_size = None, padding = None):
        super(Residual, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.downsample = downsample

        if kernel_size is None:
            kernel_size = 5
            padding = 2

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False, groups=out_channels//4)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=kernel_size, stride=1, padding=padding, bias=False, groups=1)

        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.norm1 = nn.GroupNorm(num_channels=out_channels,affine=True, num_groups=out_channels//4)
        self.norm2 = nn.GroupNorm(num_channels=out_channels,affine=True, num_groups=out_channels//4)

    def forward(self, x):

        if self.downsample is not None:
            x = self.downsample(x)

        out = x
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.relu2(out)

        out = x + out

        out = self.norm2(out)

        return out

class Residual_bottleneckgn(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample = None, kernel_size = None, padding = None):
        super(Residual_bottleneckgn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.downsample = downsample

        compression = 4

        if kernel_size is None:
            kernel_size = 5
            padding = 2

        self.conv_pre = nn.Conv1d(in_channels=in_channels, out_channels=out_channels//compression, kernel_size=1, stride=stride,
                               padding=0, bias=False)
        self.conv1 = nn.Conv1d(in_channels=out_channels//compression, out_channels=out_channels//compression, kernel_size=kernel_size, stride=1, padding=padding, bias=False, groups=1)

        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.norm1 = nn.GroupNorm(num_channels=out_channels//compression,affine=True,num_groups=4)
        self.norm2 = nn.GroupNorm(num_channels=out_channels//compression,affine=True,num_groups=4)
        self.conv_post = nn.Conv1d(in_channels=out_channels // compression, out_channels=out_channels,
                               kernel_size=1, padding=0, bias=False)
        self.norm_post = nn.GroupNorm(num_channels=out_channels,affine=True,num_groups=4)

    def forward(self, x):

        if self.downsample is not None:
            x = self.downsample(x)

        out = x
        out = self.conv_pre(out)
        out = self.norm1(out)
        out = self.relu1(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu2(out)

        out = self.conv_post(out)
        out = self.norm_post(out)

        out = x + out

        return out

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample = None, kernel_size = None, padding = None):
        super(VGGBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.downsample = downsample
        if kernel_size is None:
            kernel_size = 3
            padding = 1

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride,bias=False)

        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=1,bias=False)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.norm1 = nn.InstanceNorm1d(num_features=out_channels,affine=True)#nn.GroupNorm(num_groups=4,num_channels=out_channels)
        self.norm2 = nn.InstanceNorm1d(num_features=out_channels,affine=True)#nn.GroupNorm(num_groups=4,num_channels=out_channels)

    def forward(self, x):

        if self.downsample is not None:
            x = self.downsample(x)

        out = x
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.relu1(out)


        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)

        return out