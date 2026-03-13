import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *

import argparse
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn.functional as F
import os
from util.data_vi import data_visual, visualize_attention
from util.data import *
from util.preprocess import *

def loss_func(y_pred, y_true):
    loss = F.mse_loss(y_pred, y_true, reduction='mean')

    return loss

def test(model, dataloader):
    # test
#    loss_func = nn.MSELoss(reduction='mean')
    device = get_device()

    test_loss_list = []
    now = time.time()

    test_predicted_list = []
    test_ground_list = []
    test_labels_list = []

    t_test_predicted_list = []
    t_test_ground_list = []
    t_test_labels_list = []

    test_len = len(dataloader)

    model.eval()

    i = 0
    acu_loss = 0
    #v1,v2,v3,v4=[]
    for x, y, labels, edge_index in dataloader:
        x, y, labels, edge_index = [item.to(device).float() for item in [x, y, labels, edge_index]]
        
        with torch.no_grad():
            predicted = model(x, False).float().to(device)

            #可视化
            # predicted,v1,v2,v3,v4,v5 = model(x, edge_index)
            # predicted=predicted.float().to(device)
            # v1 = v1.cpu()
            # v2 = v2.cpu()
            # v3 = v3.cpu()
            # v4 = v4.cpu()
            # v5 = v5.cpu()

            loss = loss_func(predicted, y)
            #print("loss:",loss)

            labels = labels.unsqueeze(1).repeat(1, predicted.shape[1])

            if len(t_test_predicted_list) <= 0:
                t_test_predicted_list = predicted
                t_test_ground_list = y
                t_test_labels_list = labels
            else:
                t_test_predicted_list = torch.cat((t_test_predicted_list, predicted), dim=0)
                t_test_ground_list = torch.cat((t_test_ground_list, y), dim=0)
                t_test_labels_list = torch.cat((t_test_labels_list, labels), dim=0)

            #visualize_attention(attn, batch_idx=0, use_first_head=False)
        
        test_loss_list.append(loss.item())
        acu_loss += loss.item()
        
        i += 1

        if i % 10000 == 1 and i > 1:
            print(timeSincePlus(now, i / test_len))

    #print("loss:",loss)
    #print("acu_loss",loss)

    test_predicted_list = t_test_predicted_list.tolist()
    test_ground_list = t_test_ground_list.tolist()
    test_labels_list = t_test_labels_list.tolist()



    
    avg_loss = sum(test_loss_list)/len(test_loss_list)
    #print("avg_loss",avg_loss)

    #可视化
    # data_visual(v1)
    # data_visual(v2)
    # data_visual(v3)
    # data_visual(v4)
    # data_visual(v5)

    return avg_loss, [test_predicted_list, test_ground_list, test_labels_list]




