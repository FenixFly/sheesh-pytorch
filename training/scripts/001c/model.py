import torch
import torch.nn as nn
import gc
import torch.nn.functional as F
import pytorch_lightning as pl

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


        #self.fc = nn.Sequential(
        #    nn.Linear(in_features=number_of_channels[-1],out_features=number_of_channels[-1]//2,bias=False),
        #    nn.ReLU(),
        #    nn.Linear(in_features=number_of_channels[-1]//2,out_features=number_of_outputs,bias=True)
        #)
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
            #rate = 2**(self.depth-i-1)
            skip = skip_connections[i]#F.avg_pool3d(skip_connections[i],kernel_size=(1,int(rate),1),stride=(1,int(rate),1))

            conc = torch.cat([skip,conv],dim=1)
            conv = self.decoder_convs[i](conc)

        s = self.conv_output(conv)
        out = torch.sigmoid(s)

        return {
            'prediction': out,
        }

