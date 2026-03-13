import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *
from torch_geometric.nn import GCNConv, GATConv, EdgeConv
import math
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn, Tensor
from .graph_layer import GraphLayer
from .layers import TensorGraphConvolution, TensorGraphConvolutionFFT,TensorLinear
from util.adj_metric import *
from .my_layers import *
from .TRPCA_torch import TRPCA



class OutLayer(nn.Module):
    def __init__(self, in_num, node_num, layer_num, inter_num=512):
        super(OutLayer, self).__init__()

        modules = []

        for i in range(layer_num):
            # last layer, output shape:1
            if i == layer_num - 1:
                modules.append(nn.Linear(in_num if layer_num == 1 else inter_num, 1))
            else:
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Linear(layer_in_num, inter_num))
                modules.append(nn.BatchNorm1d(inter_num))
                modules.append(nn.ReLU())

        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x

        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0, 2, 1)
                out = mod(out)
                out = out.permute(0, 2, 1)
            else:
                out = mod(out)

        return out


class Extractor(nn.Module):
    def __init__(self, residual_channels: int, conv_channels: int, kernel_set, dilation_factor: int, dropout: float):
        super(Extractor, self).__init__()

        self.t_conv = TConv(residual_channels, conv_channels, kernel_set, dilation_factor,
                            dropout)  # Temporal Convolution Module

    def forward(self, x: Tensor):
        residual = x  # [B, F, N, T]
        # dilated convolution
        x = self.t_conv(x)

        return x


class TenGCNFFT(nn.Module):
    def __init__(self, nfeat, transm, numnodes, nhid, nchannel, nclass, dropout, batch_size):
        super(TenGCNFFT, self).__init__()

        self.gc1 = TensorGraphConvolutionFFT(nfeat, transm, numnodes, nhid, nchannel, batch_size)#W1-tow TGAT layer
        #self.gc1 = TensorGraphConvolutionFFT(nfeat, transm, numnodes, nclass, nchannel, batch_size)# W1-one TGAT layer
        #self.gch = TensorGraphConvolutionFFT(nhid, transm, numnodes, nhid, nchannel, batch_size)#W-three TGAT layers
        self.gc2 = TensorGraphConvolutionFFT(nhid, transm, numnodes, nclass, nchannel, batch_size)#W2
        self.nn3 = TensorLinear(nchannel, numnodes, 1, nclass, batch_size)#W3
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        #one TGAT layer
        #x = self.nn3(x)

        #three TGAT layers
        # x = F.relu(self.gch(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)

        #tow TGAT layers
        x = self.gc2(x, adj) #2708*7*4
        x = self.nn3(x)
        self.output_features = x
        return x

class TGDN(nn.Module):
    def __init__(self,edge_index_sets, adj_metric, node_num, dim=128, out_layer_inter_dim=256, input_dim=10, out_layer_num=1, topk=20, batch_size=32,num_subadj=8,
                 kernel_set=[2,3,6,7],dilation_exp=2,n_blocks=1,n_layers=5,gru_interval=7,svd_rank=10):
        super(TGDN, self).__init__()
        self.edge_index_sets = edge_index_sets
        self.adj_metric_sets=[]
        #self.adj_metric_sets.append(adj_metric)
        embed_dim = dim
        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)#归一化维度为embed_dim的那一维
        edge_set_num = len(edge_index_sets)
        self.dropout=0.2
        self.embedding = nn.Embedding(node_num, embed_dim)
        self.node_num=node_num
        self.slide_win=input_dim
        self.dropout = 0.3
        # self.gnn_layers = nn.ModuleList([
        #     GNNLayer(input_dim, dim, inter_dim=dim + embed_dim, heads=1) for i in range(edge_set_num)#!!!!!!!!!!!根据RTGCN进行修改
        # ])
        self.data_encode=nn.Linear(self.slide_win,embed_dim)
        self.data_feature=nn.ModuleList()
        self.num_subadj=num_subadj

        #Ver1
        for i in range(num_subadj):
            self.data_feature.append(nn.Linear(self.slide_win, embed_dim))

        self.GRU_num_layers=3
        # self.GRU=nn.GRU(input_size=1, hidden_size=1, num_layers=self.GRU_num_layers, batch_first=True)
        self.gru_hid=dim
        self.gru_interval=gru_interval
        self.linear_s2d=nn.Linear(self.slide_win, self.gru_hid)
        self.GRU=TGRU(self.gru_interval , self.gru_hid)
        self.n_layers=n_layers
        self.seq_length = input_dim
        self.svd_rank=svd_rank

        kernel_size = kernel_set[-1]

        if dilation_exp > 1:
            self.receptive_field = int(1+n_blocks*(kernel_size-1)*(dilation_exp**n_layers-1)/(dilation_exp-1))
        else:
            self.receptive_field = n_blocks * n_layers * (kernel_size - 1) + 1
        self.total_t_len = max(self.receptive_field, self.seq_length)

        # if dilation_exp > 1:
        #     rf_block = int(1 + (kernel_size - 1) * (dilation_exp ** n_layers - 1) / (dilation_exp - 1))
        # else:
        #     rf_block = n_layers * (kernel_size - 1) + 1
        rf_block=1
        self.start_conv = nn.Conv2d(1, 16, kernel_size=(1, 1))  # Conv2d(1, 16, kernel_size=(1, 1))
        dilation_factor = 1
        self.dilation = nn.ModuleList()
        self.dilateOutput=nn.ModuleList()
        for i in range(1, n_layers + 1):
            if dilation_exp>1:
                rf_size_i = int(rf_block + (kernel_size-1)*(dilation_exp**i-1)/(dilation_exp-1))
            else:
                rf_size_i = rf_block + i*(kernel_size-1)
            t_len_i = self.total_t_len - rf_size_i +1
            self.dilation.append(
                nn.Sequential(
                    Extractor(16, 16, kernel_set, dilation_factor, self.dropout),
                    nn.Linear(t_len_i, dim)
                )
            )
            # self.dilateOutput.append(
            #     nn.Linear(t_len_i,dim)
            # )
            dilation_factor *= dilation_exp

        self.transfer_conv = nn.Conv2d(16, 1, kernel_size=(1, 1))


        #self.t_conv = TConv(1, 1, kernel_set, dilation_factor, dropout)

        transform_matrix = torch.eye(node_num).cuda()
        transform_matrix=transform_matrix.unsqueeze(0).repeat(batch_size,1,1)

        self.Tensor_GCN = TenGCNFFT(
                                    #Ver1
                                    nfeat=dim,
                                    #Ver2
                                    # nfeat=self.slide_win,
                                    transm = transform_matrix,
                                    numnodes = node_num,
                                    nhid=out_layer_inter_dim,
                                    nchannel = num_subadj,
                                    nclass=dim,
                                    dropout=self.dropout,
                                    batch_size=batch_size)

        self.node_embedding = None
        self.topk = topk
        self.learned_graph = None

        self.out_layer = OutLayer(dim * edge_set_num, node_num, out_layer_num, inter_num=out_layer_inter_dim)

        self.cache_edge_index_sets = [None] * edge_set_num
        self.cache_embed_index = None

        self.dp = nn.Dropout(0.35)

        self.init_params()

    def init_params(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, data, org_edge_index):
        x = data.clone().detach()
        batch_x = data.clone().detach()
        batch_x1 = data.clone().detach()
        batch_x2 = data.clone().detach()
        edge_index_sets = self.edge_index_sets

        device = data.device
        # print(device)

        batch_num, node_num, all_feature = x.shape
        # torch.size=[32,127,5]#slide_win=5

        #x = x.view(-1, all_feature).contiguous()

        #Ver1
        x_feature1=self.data_feature[0](x)
        x_feature2 = self.data_feature[1](x)
        x_feature3 = self.data_feature[2](x)
        #Ver2
        # x_feature1 = x
        # x_feature2 = x
        # x_feature3 = x


        Tgcn_outs = []
        for i, edge_index in enumerate(edge_index_sets):
            subadj = []
            fea = []

###########根据Embeding来构造adj_m
            all_embeddings = self.embedding(torch.arange(node_num).to(device))
            weights_arr = all_embeddings.detach().clone()
            #all_embeddings = all_embeddings.repeat(batch_num, 1)
            all_embeddings = all_embeddings.repeat(batch_num, 1,1)
            #fea.append(all_embeddings)

            #fea.append(x_feature1)#TGAT-Linear

            # print(all_embeddings.shape)
            embedding_adj_m=Embeding_adj(self.node_num, all_embeddings, weights_arr, device, self.topk, batch_num)#[batch_size, node_num, node_num]
            subadj.append(embedding_adj_m)
            #可视化
            v1=weights_arr

############根据data-value来构造adj_m
            encode_data_x=self.data_encode(batch_x)
            #fea.append(encode_data_x)

            #fea.append(x_feature2)#TGAT-Linear

            # print(encode_data_x.shape)
            datavalue_adj_m=Data_adj(self.node_num, encode_data_x, device, self.topk, batch_num)
            subadj.append(datavalue_adj_m)
            #可视化
            v2=encode_data_x[10]

############利用GRU获取每个节点的temporal信息，并以此来构造adj_m
            tmp2=batch_x1
            states_dy=self.linear_s2d(tmp2)
            for i_t in range(0,all_feature,self.gru_interval):
                x_i = batch_x1[..., i_t:min(i_t + self.gru_interval, all_feature)]
                #x_i.shape=[batch_size,node_num,gru_interval]
                #states_dy.shape=[batch_size,node_num,gru_hid/dim]
                states_dy=self.GRU(x_i,states_dy)
            GRU_adj_input=states_dy
            #GRU_adj_input.shape=[batch_size,node_num,dim]
            #fea.append(GRU_adj_input)

            #fea.append(x_feature3)#TGAT-Linear

            GRU_adj_m = Data_adj(self.node_num, GRU_adj_input, device, self.topk, batch_num)
            subadj.append(GRU_adj_m)
            #可视化
            v3=GRU_adj_input[10]

############利用 dilated inception layer来捕获 multi-scale representations
            flag = 0
            if self.seq_length < self.receptive_field:
                batch_x2 = F.pad(batch_x2, (self.receptive_field - self.seq_length, 0, 0, 0), mode='replicate')
            tmp1=batch_x2.unsqueeze(1)
            for layer in self.dilation:
                #Ver1
                x_feature=self.data_feature[flag+self.num_subadj-self.n_layers](x)#TGAT-Linear
                #Ver2
                #x_feature=x

                tmp1 = self.start_conv(tmp1)
                tmp1= layer[0](tmp1)
                tmp1=self.transfer_conv(tmp1)
                flag += 1
                # print("flagtmp1:",flag," ",tmp1.shape)
                dilate_adj_input=torch.squeeze(tmp1)
                #dilate_adj_input=dilate_adj_input.unsqueeze(0)#!!!!!!!!!!!!!!!!!!!!!!!!!lalalaalla
                if flag==1:#可视化
                    v4 = dilate_adj_input[10]
                    tmp2 = layer[1](dilate_adj_input)
                if flag==self.n_layers:
                    dilate_adj_input=dilate_adj_input.unsqueeze(2)

                dilate_adj_m=Data_adj(self.node_num, dilate_adj_input,device,self.topk,batch_num)

                # print("flagtmp2:", flag, " ", tmp2.shape)
                #fea.append(tmp2)

                #fea.append(x_feature)#TGAT-Linear

                subadj.append(dilate_adj_m)

            for ii in range(self.num_subadj):
                #fea.append(GRU_adj_input)#TGAT-GRU
                #fea.append(tmp2)#TGAT-DIL
                fea.append(encode_data_x)#TGAT-Embed

            adj2 = torch.stack(subadj, dim=3)
            features2 = torch.stack(fea, dim=3)

########### T-SVD
            # model0 = TRPCA()
            # adj2 = adj2.cuda()
            # batch_L=[]
            # for i in range(batch_num):
            #     adj3=adj2[i,...]
            #     L = model0.T_SVD(adj3, self.svd_rank)
            #     batch_L.append(L)
            # LL= torch.tensor([item.cpu().detach().numpy() for item in batch_L]).cuda()
            # LL=torch.nn.functional.softmax(LL,dim=2)
            model0 = TRPCA()
            adj2 = adj2.cuda()
            batch_L = []
            batch_fea = []
            for i in range(batch_num):
                adj3 = adj2[i, ...]
                #features3 = features2[i, ...]
                L = model0.T_SVD(adj3, self.svd_rank)
                #F1 = model0.T_SVD(features3, self.svd_rank)
                batch_L.append(L)
                #batch_fea.append(F1)
            LL = torch.tensor([item.cpu().detach().numpy() for item in batch_L]).cuda()
            #FF = torch.tensor([item.cpu().detach().numpy() for item in batch_fea]).cuda()
            LL = torch.nn.functional.softmax(LL, dim=2)
            #FF = torch.nn.functional.softmax(FF, dim=2)

########### TGCN
            output=self.Tensor_GCN(features2, LL)
            # adj2=torch.nn.functional.softmax(adj2, dim=2)
            # output = self.Tensor_GCN(features2, adj2)
            v5=output[10]
            Tgcn_outs.append(output)

        x = torch.cat(Tgcn_outs, dim=2)
        #x.shape=[batch,node_num,dim]
        indexes = torch.arange(0, node_num).to(device)
        out = torch.mul(x, self.embedding(indexes))
        out = out.permute(0, 2, 1)
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0, 2, 1)

        out = self.dp(out)
        out = self.out_layer(out)
        out = out.view(-1, node_num)

        return out

        #可视化
        #return out, v1, v2, v3, v4, v5

















