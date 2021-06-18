import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from io import StringIO
import pandas as pd
import pytorch_lightning as pl
import gc
import numpy as np
import random
import collections

def pad_to_shape(image, shape, mode='minimum'):
    diff = np.array(shape) - np.array(image.shape)

    diff[np.isnan(diff)] = 0
    diff = diff.astype(np.int32)

    pad_left = diff // 2
    pad_right = diff - pad_left
    padded = np.pad(image, pad_width=tuple(zip(pad_left, pad_right)), mode=mode)
    return padded, (pad_left, pad_right)

def revert_pad(image, padding, dim:list=None):
    pad_left, pad_right = padding

    if dim is None:
        dim = range(len(image.shape))

    start = pad_left
    end = [s - r for s, r in zip(image.shape,pad_right)]

    index = tuple([slice(s, e) if idx in dim else slice(None) for idx,(s, e) in enumerate(zip(start, end))])

    return image[index]

def closest_to_k(n,k=8):
    if n % k == 0:
        return n
    else:
        return ((n // k) + 1)*k

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

class Model(pl.LightningModule):

    def __init__(self,model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

class Linear(nn.Module):
    def __init__(self, scale):
        super(Linear, self).__init__()
        self.scale = scale

    def forward(self, x):
        out = F.interpolate(x,scale_factor=self.scale,mode='linear',align_corners=False)
        return out


class CustomNet(nn.Module):
    def __init__(self, depth, encoder_layers, number_of_channels, number_of_outputs, block):
        super(CustomNet, self).__init__()
        print('CustomNet {}'.format(number_of_channels))
        self.encoder_layers = encoder_layers
        self.block=block
        self.number_of_outputs = number_of_outputs
        self.number_of_channels = number_of_channels
        self.depth = depth

        self.encoder_convs = nn.ModuleList()
        self.upsampling = nn.ModuleList()
        self.decoder_convs = nn.ModuleList()

        conv_first_list = [nn.Conv1d(in_channels=1, out_channels=self.number_of_channels[0], kernel_size=3, stride=1,
                                   padding=1, bias=True),
                           nn.BatchNorm1d(num_features=self.number_of_channels[0])
                           ]

        for i in range(self.encoder_layers[0]):
            conv_first_list.append(
                self.block(in_channels=self.number_of_channels[0], out_channels=self.number_of_channels[0],
                              stride=1))


        self.conv_first = nn.Sequential(*conv_first_list)
        self.conv_output = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv1d(in_channels=self.number_of_channels[0],out_channels=number_of_outputs,kernel_size=5,
                                     stride=1,padding=2,bias=True,groups=1),
        )

        self.sigmoid = nn.Sigmoid()


        self.construct_encoder_convs(depth=depth,number_of_channels=number_of_channels)
        self.construct_upsampling_convs(depth=depth,number_of_channels=number_of_channels)
        self.construct_decoder_convs(depth=depth,number_of_channels=number_of_channels)

    def _make_encoder_layer(self, in_channels, channels, blocks, block, stride=1, ds_kernel = (2,2), ds_stride = (2,2)):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.AvgPool1d(kernel_size=ds_kernel,stride=ds_stride),
                nn.Conv1d(in_channels=in_channels, out_channels=channels, kernel_size=1,bias=False)
            )

        layers = []
        layers.append(block(in_channels=channels, out_channels=channels, stride=1, downsample=downsample))

        for _ in range(1, blocks):
            layers.append(block(in_channels=channels, out_channels=channels, stride=1))

        return nn.Sequential(*layers)


    def construct_encoder_convs(self, depth, number_of_channels):
        for i in range(depth - 1):
            conv = self._make_encoder_layer(in_channels=number_of_channels[i], channels=number_of_channels[i + 1],
                                            blocks=self.encoder_layers[i + 1], stride=2, block=self.block, ds_kernel=2, ds_stride=2)
            self.encoder_convs.append(conv)


    def construct_decoder_convs(self, depth, number_of_channels):

        for i in range(depth):
            conv_list = []
            for j in range(self.encoder_layers[i]):
                conv_list.append(self.block(in_channels=number_of_channels[i], out_channels=number_of_channels[i], stride=1,
                                            kernel_size=3, padding = 1))

            dec_conv = nn.Sequential(
                *conv_list,
                nn.Dropout(p=0.2),
            )

            conv= nn.Sequential(
                nn.Conv1d(in_channels=2*number_of_channels[i], out_channels=number_of_channels[i], kernel_size=3,stride=1,padding=1, bias=False),
                nn.BatchNorm1d(num_features=number_of_channels[i], affine=True),
                dec_conv
            )
            self.decoder_convs.append(conv)

    def construct_upsampling_convs(self, depth, number_of_channels):
        for i in range(depth-1):
            conv =  nn.Sequential(
                Linear(scale=2),
                nn.Conv1d(in_channels=number_of_channels[i+1], out_channels=number_of_channels[i], kernel_size=1,stride=1,padding=0, bias=False),

            )
            self.upsampling.append(conv)

    def forward(self, x):
        input = x['signal']
        skip_connections = []

        N, C, W = input.shape
        conv = self.conv_first(input)

        for i in range(self.depth - 1):
            skip_connections.append(conv)
            conv = self.encoder_convs[i](conv)

        for i in reversed(range(self.depth - 1)):
            conv = self.upsampling[i](conv)
            skip = skip_connections[i]

            conc = torch.cat([skip,conv],dim=1)
            conv = self.decoder_convs[i](conc)

        s = self.conv_output(conv)
        out = torch.sigmoid(s)

        return {
            'prediction': out,
        }

class Transform:
    def __init__(self, transform_keys:list):
        self.transform_keys = transform_keys

    def __call__(self, data:dict):
        raise NotImplementedError()

class Compose:
    def __init__(self, transforms:list):
        self.transforms = transforms

    def __call__(self, data:dict):
        results = data
        for t in self.transforms:
            results = t(data)
        return results

class DualCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        results = args
        for t in self.transforms:
            results = t(*results)
        return results

class PadToRate(Transform):
    def __init__(self, transform_keys: list, rate:list, dim:list=None):
        super(PadToRate, self).__init__(transform_keys)
        self.rate = np.array(rate)
        self.dim = dim

    def pad_to_rate(self,array,key):
        data_shape = array.shape
        if self.dim is None:
            self.dim = list(range(len(data_shape)))

        new_shape = [closest_to_k(i, r) if idx in self.dim else i for idx, (i, r) in
                     enumerate(zip(data_shape, self.rate))]
        new_shape = tuple(new_shape)

        padded, (pad_left, pad_right) = pad_to_shape(array, new_shape)


        desc = {}
        desc['shape_old' + key] = data_shape
        desc['shape_new' + key] = new_shape
        desc['padding' + key] = (pad_left, pad_right)
        return padded, desc

    def __call__(self, data: dict):

        for key in self.transform_keys:

            if isinstance(data[key],dict):
                for subkey in data[key]:
                    array, desc = self.pad_to_rate(data[key][subkey], key)
                    data[key][subkey] = array
                    data.update(desc)
            else:
                array, desc = self.pad_to_rate(data[key],key)
                data[key] = array
                data.update(desc)

        return data

class Normalization(Transform):
    def __init__(self, transform_keys:list, mean, std):
        super(Normalization, self).__init__(transform_keys)
        self.mean = mean
        self.std = std

    def __call__(self, data:dict):
        for key in self.transform_keys:
            data[key] = (data[key] - self.mean)/(self.std)

        return data

class ToTensorDict(Transform):
    def __init__(self, transform_keys: list):
        super(ToTensorDict, self).__init__(transform_keys)

    def __call__(self, data: dict):
        for key in self.transform_keys:


            if isinstance(data[key],dict):
                for subkey in data[key]:
                    data[key][subkey] = torch.from_numpy(data[key][subkey]).float()

            else:
                data[key] = torch.from_numpy(data[key]).float()

        return data

class ToTensor:
    def __call__(self, *args):

        results = []

        for a in args:
            img = torch.from_numpy(a).float()
            results.append(img)

        return tuple(results)


layers = [2, 2, 4]
n_outputs = 1
number_of_channels = [int(16 * 1 * 2 ** i) for i in range(0, len(layers))]
arch = CustomNet(depth=len(layers), encoder_layers=layers, number_of_channels=number_of_channels,
                           number_of_outputs=n_outputs, block=Residual_bottleneck)
params = {'model': arch}


def make_output(df):
    res = ''

    # Make header
    for i in range(len(df.columns.values)-1):
        res += str(df.columns.values[i])
        res += ','
    res += str(df.columns.values[-1])
    
    #Make table
    for i in range(df.shape[0]):
        line_list = df.iloc[i].to_numpy()
        line_str = '\n'
        for j in range(2):
            line_str += str(line_list[j])
            line_str += ','
        line_str += str(line_list[-1])
        res += line_str

    return res

def handler(context, event):

    # load model & create transforms
    data_transform_val = Compose([
        Normalization(transform_keys=['signal'], mean=617.7666, std=287.5535),
        PadToRate(transform_keys=['signal', ], rate=[1, 64]),
        ToTensorDict(transform_keys=['signal', ])
    ])
    model_trainer = Model.load_from_checkpoint("/tmp/model/model.ckpt",**params,map_location=torch.device('cpu'))
    model_trainer.eval()
    torch.no_grad()

    # Read data from text body
    f = StringIO( event.body.decode('utf-8'))
    
    # Read patient data
    df = pd.read_csv(f, sep=",").sort_values(by='time', axis=0, ascending=True)
    df['y'] = 0

    # Construct batch
    batch = {'signal': df['x'].to_numpy()[None]}
    batch = data_transform_val(batch)
    batch['signal'] = batch['signal'][None] #add batch dimension

    # Predict
    output = model_trainer(batch)

    # Revert padding from transform
    output = output['prediction'][0].cpu().detach().numpy()
    pad_left, pad_right = batch['paddingsignal']
    output = revert_pad(output, (pad_left, pad_right), dim=[1])[0]

    # Add result to dataframe 
    df['y'] = output

    # convert csv to text to sent to webserver
    result_json = make_output(df)
    
    return context.Response(body=result_json,
                            headers={},
                            content_type='text/plain',
                            status_code=200)