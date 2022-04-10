##################################
# EMBER2
# Inference pipeline for 2D structure prediction using language models
#
# Authors: Konstantin Weißenow, Michael Heinzinger
##################################

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, dilation=1, block_width=128, rezero=False):
        super(BasicBlock, self).__init__()

        half_width = int(block_width * 0.5)

        self.nonlinearity = nn.ELU(inplace=True)

        self.inputBatchNorm = nn.BatchNorm2d(block_width)
        self.project_down = nn.Conv2d(block_width, half_width, kernel_size=1, stride=1, bias=True)
        self.intermediateBatchNorm1 = nn.BatchNorm2d(half_width)
        self.convolution = nn.Conv2d(half_width, half_width, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=True)
        self.intermediateBatchNorm2 = nn.BatchNorm2d(half_width)
        self.project_up = nn.Conv2d(half_width, block_width, kernel_size=1, stride=1, bias=True)

        self.skip_scale = 1
        if rezero:
            self.skip_scale = nn.Parameter(torch.ones((64, 64)), requires_grad=True)

    def forward(self, x):
        intermediate = self.inputBatchNorm(x)
        intermediate = self.nonlinearity(intermediate)

        intermediate = self.project_down(intermediate)
        intermediate = self.intermediateBatchNorm1(intermediate)
        intermediate = self.nonlinearity(intermediate)

        intermediate = self.convolution(intermediate)
        intermediate = self.intermediateBatchNorm2(intermediate)
        intermediate = self.nonlinearity(intermediate)

        intermediate = self.project_up(intermediate)

        return intermediate + self.skip_scale * x



class DeepDilated(nn.Module):
    def __init__(self, input_channels_2d=443, input_channels_1d=0, layers=220, out_channels=42, block_width=128, rezero=False):
        super(DeepDilated, self).__init__()

        self.block_width = block_width

        self.inputChannels1d = input_channels_1d
        self.inputChannels2d = input_channels_2d

        self.nonlinearity = nn.ELU(inplace=True)

        self.inputCompression2d = nn.Conv2d(self.inputChannels2d, block_width, kernel_size=1, stride=1, bias=True)
        if self.inputChannels1d > 0:
            self.inputCompression1d = nn.Conv1d(self.inputChannels1d, int(block_width * 0.5), kernel_size=1, stride=1, padding=0, bias=True)
            self.combined_compression = nn.Sequential(
                nn.BatchNorm2d(block_width*2),
                nn.ELU(inplace=True),
                nn.Conv2d(block_width*2, block_width, kernel_size=1, stride=1, padding=0, bias=True)
            )

        layer_list = []
        dilations = [1, 2, 4, 8]
        for i in range(layers):
            layer_list.append(BasicBlock(dilation=dilations[i % 4], block_width=block_width, rezero=rezero))
        self.main_resnet = nn.Sequential(*layer_list)

        self.output_batchnorm = nn.BatchNorm2d(block_width)
        self.output_convolution = nn.Conv2d(block_width, out_channels, kernel_size=1, stride=1, bias=True)

    def forward(self, input_2d, input_1d):
        intermediate = self.inputCompression2d(input_2d)

        if self.inputChannels1d > 0:
            input_1d_x = input_1d[:, :self.inputChannels1d, :]
            input_1d_y = input_1d[:, self.inputChannels1d:, :]
            intermediate_x = self.inputCompression1d(input_1d_x)
            intermediate_y = self.inputCompression1d(input_1d_y)

            # transform 1d input and concatenate
            a = intermediate_x.unsqueeze(2).repeat(1, 1, 64, 1)
            b = intermediate_y.unsqueeze(3).repeat(1, 1, 1, 64)
            intermediate_1d = torch.cat((a, b), dim=1)
            intermediate = torch.cat((intermediate, intermediate_1d), dim=1)
            intermediate = self.combined_compression(intermediate)

        # Main ResNet pipeline
        intermediate = self.main_resnet(intermediate)

        # Finish with batch norm, non-linearity and final convolution to distance bins
        intermediate = self.output_batchnorm(intermediate)
        intermediate = self.nonlinearity(intermediate)
        return self.output_convolution(intermediate), [], []
