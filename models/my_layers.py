# import os
#
# import numpy as np
#import torch
# import matplotlib.pyplot as plt
# import torch.nn as nn
# import time
# from util.time import *
# from util.env import *
# from torch_geometric.nn import GCNConv, GATConv, EdgeConv
# import math
import torch.nn.functional as F
# from torch.autograd import Variable
from torch import nn, Tensor
# from .graph_layer import GraphLayer
# from .layers import TensorGraphConvolution, TensorGraphConvolutionFFT,TensorLinear
from util.adj_metric import *

class Dilated_Inception(nn.Module):
    def __init__(self, cin, cout, kernel_set, dilation_factor=2):
        super(Dilated_Inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = kernel_set
        cout = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin,cout,(1,kern),dilation=(1,dilation_factor)))

    def forward(self,input):
        #input [B, D, N, T]
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))

        for i in range(len(self.kernel_set)):
            x[i] = x[i][...,-x[-1].size(3):]

#            print(x[-1].size(3))
#            print(x[i].shape)
        x = torch.cat(x,dim=1)
        # print(x.shape)
        # os.system("pause")
        return x

class TConv(nn.Module):
    def __init__(self, residual_channels: int, conv_channels: int, kernel_set, dilation_factor: int, dropout:float):
        super(TConv, self).__init__()
        self.filter_conv = Dilated_Inception(residual_channels, conv_channels,kernel_set, dilation_factor)
        self.gate_conv = Dilated_Inception(residual_channels, conv_channels, kernel_set, dilation_factor)
        self.dropout = dropout

    def forward(self, x: Tensor):
        _filter = self.filter_conv(x)
        filter = torch.tanh(_filter)
        _gate = self.gate_conv(x)
        gate = torch.sigmoid(_gate)
        x = filter * gate
        #print(x.shape)
        x = F.dropout(x, self.dropout, training=self.training)
        return x

class normal_conv(nn.Module):
    def __init__(self, dg_hidden_size):
        super(normal_conv, self).__init__()
        self.fc1 = nn.Linear(dg_hidden_size, 1)#Linear(in_features=20, out_features=1, bias=True)
        self.fc2 = nn.Linear(dg_hidden_size * 2, dg_hidden_size)#Linear(in_features=40, out_features=20, bias=True)

    def forward(self, y, shape):
        support = self.fc1(torch.relu(self.fc2(y))).reshape(shape)
        #support.shape=[16,137,137]
        return support

class TGRU(nn.Module):
    def __init__(self, input_size: int, dg_hidden_size: int):
        super(TGRU, self).__init__()
        #######GRU
        self.rz_gate = nn.Linear(input_size + dg_hidden_size, dg_hidden_size * 2)#将reset gate和update gate合二为一，所以输出维度为dg_hidden_size * 2
        self.dg_hidden_size = dg_hidden_size
        self.h_candidate = nn.Linear(input_size + dg_hidden_size, dg_hidden_size)

        self.conv = normal_conv(dg_hidden_size)#######MLPe
        self.conv2 = normal_conv(dg_hidden_size)#######MLPm

    def forward(self, inputs: Tensor, states):
        """
        :param inputs: inputs to cal dynamic relations   [B,N,C]#[16,137,16]
        :param states: recurrent state [B, N,C]#[16,137,20]
        :return:   states[B,N,C]
        """
        ######GRU
        b,n,c = states.shape
        r_z = torch.sigmoid(self.rz_gate(torch.cat([inputs, states], -1)))
        r, z = r_z.split(self.dg_hidden_size, -1)
        h_ = torch.tanh(self.h_candidate(torch.cat([inputs, r * states], -1)))
        new_state = z * states + (1 - z) * h_

        return new_state

