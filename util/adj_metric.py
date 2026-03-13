import torch
from torch import nn
import numpy as np
import torch_geometric as tg
import os

def normalize_adj_tensor(adj):
    """Normalize adjacency tensor matrix.
    """
    device = adj.device
    mx = adj + torch.eye(adj.shape[0]).to(device)
    rowsum = mx.sum(1)
    r_inv = rowsum.pow(-1/2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv @ mx
    mx = mx @ r_mat_inv
    return mx

def get_batch_edge_index(org_edge_index, batch_num, node_num):
    # org_edge_index:(2, edge_num)
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1,batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i*edge_num:(i+1)*edge_num] += i*node_num

    return batch_edge_index.long()

def edge2adj(node_num, batch_edge_index, device, batch_size, org_edge_num):#太慢了
    adj_m=torch.zeros(size=(batch_size,node_num,node_num)).to(device)

    for i in range(batch_size):#这个循环要改进一下
        for j in range(org_edge_num):
            adj_m[i][batch_edge_index[1][j+i*org_edge_num]-i*node_num][batch_edge_index[0][j+i*org_edge_num]-i*node_num]=1
    return adj_m

def Attention(adj_m, node_f):
    query=node_f
    key=node_f
    key_T=torch.transpose(key,1,2)
    W=torch.bmm(query,key_T)
    #adj_W=torch.nn.functional.softmax(torch.mul(adj_m,W),dim=-1)
    adj_W=torch.mul(adj_m,W)
    return adj_W

def Embed_Attention(adj_m, embed):#尝试用additive_Attention
    query=embed
    key=embed
    key_T=torch.transpose(key,0,1)
    W=torch.mm(query,key_T)
    #adj_W=torch.nn.functional.softmax(torch.mul(adj_m,W),dim=-1)
    adj_W = torch.mul(adj_m, W)
    return adj_W

def Embeding_adj(node_num, embeding, weights_arr, device, topk, batch_size):
    weights = weights_arr.view(node_num, -1)
    cos_ji_mat = torch.matmul(weights, weights.T)
    normed_mat = torch.matmul(weights.norm(dim=-1).view(-1, 1), weights.norm(dim=-1).view(1, -1))
    cos_ji_mat = cos_ji_mat / normed_mat

    topk_num = topk
    topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]
    gated_i = torch.arange(0, node_num).T.unsqueeze(1).repeat(1, topk_num).flatten().to(device).unsqueeze(0)
    gated_j = topk_indices_ji.flatten().unsqueeze(0)
    gated_edge_index = torch.cat((gated_j, gated_i), dim=0)
    #print(gated_edge_index.shape)#[2, topk*node_num]
    org_edge_num=len(gated_edge_index[0])
    batch_gated_edge_index=get_batch_edge_index(gated_edge_index,batch_size,node_num)

    batch = torch.LongTensor(np.arange(batch_size).reshape(batch_size, 1)).to(device).repeat(1, node_num).flatten()

    adj_m = tg.utils.to_dense_adj(batch_gated_edge_index, batch)

    #adj_m=edge2adj(node_num, batch_gated_edge_index, device,batch_size,org_edge_num)
    adj_m_W=Embed_Attention(adj_m,weights_arr)

    return adj_m_W

def Data_adj(node_num, batch_x, device, topk, batch_size):

    cos_ji_mat = torch.bmm(batch_x, torch.permute(batch_x,(0,2,1)))
    normed_mat = torch.bmm(batch_x.norm(dim=-1).view(batch_size,-1, 1), batch_x.norm(dim=-1).view(batch_size,1, -1))#################
    cos_ji_mat = cos_ji_mat / normed_mat
    cos_ji_mat = torch.where(torch.isnan(cos_ji_mat), torch.full_like(cos_ji_mat, 0), cos_ji_mat)

    topk_num = topk
    org_edge_num = node_num * topk_num
    topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]
    gated_i = torch.arange(0, node_num).T.unsqueeze(1).repeat(1, topk_num).flatten().to(device).unsqueeze(0)
    gated_i=gated_i.repeat(1,batch_size)

    gated_j = topk_indices_ji.flatten().unsqueeze(0)

    gated_edge_index = torch.cat((gated_i, gated_j), dim=0)
    for i in range(batch_size):
        gated_edge_index[:, i*org_edge_num:(i+1)*org_edge_num] += i*node_num

    batch=torch.LongTensor(np.arange(batch_size).reshape(batch_size,1)).to(device).repeat(1,node_num).flatten()

    adj_m = tg.utils.to_dense_adj(gated_edge_index, batch)

    adj_m_W = Attention(adj_m, batch_x)

    return adj_m_W



def GRU_adj(node_num, GRU_adj_input, device, topk, batch_size):
    cos_ji_mat = torch.bmm(GRU_adj_input, torch.permute(GRU_adj_input, (0, 2, 1)))
    normed_mat = torch.bmm(GRU_adj_input.norm(dim=-1).view(batch_size, -1, 1),
                           GRU_adj_input.norm(dim=-1).view(batch_size, 1, -1))  #################
    cos_ji_mat = cos_ji_mat / normed_mat
    cos_ji_mat = torch.where(torch.isnan(cos_ji_mat), torch.full_like(cos_ji_mat, 0), cos_ji_mat)

    topk_num = topk
    org_edge_num = node_num * topk_num
    topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]
    gated_i = torch.arange(0, node_num).T.unsqueeze(1).repeat(1, topk_num).flatten().to(device).unsqueeze(0)
    gated_i = gated_i.repeat(1, batch_size)

    gated_j = topk_indices_ji.flatten().unsqueeze(0)
    gated_edge_index = torch.cat((gated_j, gated_i), dim=0)
    for i in range(batch_size):
        gated_edge_index[:, i * org_edge_num:(i + 1) * org_edge_num] += i * node_num
    # print(gated_edge_index.shape)#[2, topk*node_num]

    # batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_size, node_num)
    adj_m = edge2adj(node_num, gated_edge_index, device, batch_size, org_edge_num)
    adj_m_W = Attention(adj_m, GRU_adj_input)

    return adj_m_W

def Dilate_adj(node_num, Dilate_adj_input, device, topk, batch_size):
    cos_ji_mat = torch.bmm(Dilate_adj_input, torch.permute(Dilate_adj_input, (0, 2, 1)))
    normed_mat = torch.bmm(Dilate_adj_input.norm(dim=-1).view(batch_size, -1, 1),
                           Dilate_adj_input.norm(dim=-1).view(batch_size, 1, -1))  #################
    cos_ji_mat = cos_ji_mat / normed_mat
    cos_ji_mat = torch.where(torch.isnan(cos_ji_mat), torch.full_like(cos_ji_mat, 0), cos_ji_mat)

    topk_num = topk
    org_edge_num = node_num * topk_num
    topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]
    gated_i = torch.arange(0, node_num).T.unsqueeze(1).repeat(1, topk_num).flatten().to(device).unsqueeze(0)
    gated_i = gated_i.repeat(1, batch_size)

    gated_j = topk_indices_ji.flatten().unsqueeze(0)
    gated_edge_index = torch.cat((gated_j, gated_i), dim=0)
    for i in range(batch_size):
        gated_edge_index[:, i * org_edge_num:(i + 1) * org_edge_num] += i * node_num
    # print(gated_edge_index.shape)#[2, topk*node_num]

    # batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_size, node_num)
    adj_m = edge2adj(node_num, gated_edge_index, device, batch_size, org_edge_num)
    adj_m_W = Attention(adj_m, Dilate_adj_input)


    return adj_m_W

