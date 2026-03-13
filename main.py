import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset

from util.env import get_device, set_device
from util.preprocess import build_loc_net, construct_data
from util.net_struct import get_feature_map, get_fc_graph_struc

from datasets.TimeDataset import TimeDataset

from util.adj_metric import edge2adj

from models.VST_FusionNet import VST_FusionNet

from train import train
from test  import test
from evaluate import get_err_scores, get_best_performance_data, get_val_performance_data, get_full_err_scores,get_val_performance_data_2
import sys
from datetime import datetime

import os
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import json
import random

class Main():
    def __init__(self,train_config, env_config, debug=False):
        self.train_config=train_config
        self.env_config=env_config

        self.datestr = None

        dataset=self.env_config["dataset"]
        batch_size=self.env_config["batch"]
        train_orig = pd.read_csv(f'./data/{dataset}/train.csv', sep=',', index_col=0)
        test_orig = pd.read_csv(f'./data/{dataset}/test.csv', sep=',', index_col=0)


        train, test = train_orig, test_orig

        if 'attack' in train.columns:
            train = train.drop(columns=['attack'])

        feature_map = get_feature_map(dataset)#获取节点名字
        fc_struc = get_fc_graph_struc(dataset)#一开始 graph is 'fully-connect'，这个fc_struc是个map

        set_device(env_config['device'])
        self.device = get_device()

        fc_edge_index = build_loc_net(fc_struc, list(train.columns), feature_map=feature_map)#获得适合于torch_geometric的边结构
        fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)
        self.feature_map = feature_map
        #print(type(train))#<class 'pandas.core.frame.DataFrame'>
        train_dataset_indata = construct_data(train, feature_map, labels=0)#将train数据转换为列表list
        test_dataset_indata = construct_data(test, feature_map, labels=test.attack.tolist())
        #print(type(train_dataset_indata))#<class 'list'>

        cfg = {
            'slide_win': train_config['slide_win'],
            'slide_stride': train_config['slide_stride'],
        }
        train_dataset = TimeDataset(train_dataset_indata, fc_edge_index, batch_size=train_config['batch'], mode='train', config=cfg)
        test_dataset = TimeDataset(test_dataset_indata, fc_edge_index, batch_size=train_config['batch'], mode='test', config=cfg)
        #print(train_dataset.x.shape)#[total_time_len-slide_win,node_num,slide_win]
        #print(train_dataset.y.shape)#[total_time_len-slide_win,node_num]
        #print(test_dataset.x.shape)
        aa=1
        # print(train_orig.shape)
        # if train_orig.shape[0] % batch_size != 0:
        #     yu = train_orig.shape[0] % batch_size
        #     train_orig = train_orig[yu:, :]
        #
        # print(train_orig.shape[0])
        # os.system("pause")

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        train_dataloader, val_dataloader = self.get_loaders(train_dataset, train_config['seed'], train_config['batch'],
                                                            val_ratio=train_config['val_ratio'])
        self.train_dataloader = train_dataloader
        #print(len(train_dataloader))#(total_time_len-slide_win)*(1-val_ratio)/batch_size
        self.val_dataloader = val_dataloader
        self.test_dataloader = DataLoader(test_dataset, batch_size=train_config['batch'],
                                          shuffle=False, num_workers=0)



        edge_index_sets = []
        edge_index_sets.append(fc_edge_index)
        #print(type(fc_edge_index))
        #构造邻接矩阵
        self.node_num=len(feature_map)
        #adj_m=torch.zeros(size=(self.node_num,self.node_num))
        adj_m=edge2adj(self.node_num, fc_edge_index, self.device, 1, len(fc_edge_index[0]))#这个有什么用？

        self.model = VST_FusionNet(batch=train_config['batch'],node_num=self.node_num,hidden_dim=train_config['hidden_dim'],
                                   slide_win=train_config['slide_win'],ts_patch=train_config['ts_patch'],
                                   slide_stride=train_config['slide_stride'],tsf_dim=train_config['tsf_dim'],
                                   use_stft=train_config['use_stft'],delay=train_config['delay'],
                                   embedding=train_config['embedding'],img_size=train_config['img_size'],
                                   patch_size=train_config['patch_size'],embed_dim=train_config['embed_dim'],
                                   en_depth=train_config['en_depth'],en_num_heads=train_config['en_num_heads'],
                                   num_query_token=train_config['num_query_token'],freeze_vit=train_config['freeze_vit'],
                                   out_layer_num=train_config['out_layer_num'],out_layer_inter_dim=train_config['out_layer_inter_dim'],
                                   device=self.device).to(self.device)

    def run(self):

        if len(self.env_config['load_model_path']) > 0:
            model_save_path = self.env_config['load_model_path']
        else:
            model_save_path = self.get_save_path()[0]

            self.train_log = train(self.model, model_save_path,
                                   config=train_config,
                                   train_dataloader=self.train_dataloader,
                                   val_dataloader=self.val_dataloader,
                                   feature_map=self.feature_map,
                                   test_dataloader=self.test_dataloader,
                                   test_dataset=self.test_dataset,
                                   train_dataset=self.train_dataset,
                                   dataset_name=self.env_config['dataset']
                                   )
        # test
        self.model.load_state_dict(torch.load(model_save_path))
        best_model = self.model.to(self.device)

        _, self.test_result = test(best_model, self.test_dataloader)
        _, self.val_result = test(best_model, self.val_dataloader)

        self.get_score(self.test_result, self.val_result)


    def get_loaders(self, train_dataset, seed, batch, val_ratio=0.1):
        dataset_len = int(len(train_dataset))
        yu=(dataset_len * (1 - val_ratio))%batch
        yu2=(dataset_len * val_ratio)%batch
        train_use_len = int(dataset_len * (1 - val_ratio)-yu)
        val_use_len = int(dataset_len * val_ratio-yu2)
        val_start_index = random.randrange(train_use_len)
        indices = torch.arange(dataset_len)

        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]])
        train_subset = Subset(train_dataset, train_sub_indices)

        val_sub_indices = indices[val_start_index:val_start_index+val_use_len]
        val_subset = Subset(train_dataset, val_sub_indices)


        train_dataloader = DataLoader(train_subset, batch_size=batch,
                                shuffle=True)

        val_dataloader = DataLoader(val_subset, batch_size=batch,
                                shuffle=False)

        return train_dataloader, val_dataloader

    def get_score(self, test_result, val_result):
        # print(test_result.shape)
        feature_num = len(test_result[0][0])
        np_test_result = np.array(test_result)
        np_val_result = np.array(val_result)

        test_labels = np_test_result[2, :, 0].tolist()

        test_scores, normal_scores = get_full_err_scores(test_result, val_result)
        # test_scores.shape=[38,2784]

        top1_best_info = get_best_performance_data(test_scores, test_labels, topk=1)
        top1_val_info = get_val_performance_data(test_scores, normal_scores, test_labels, topk=1)
        top1_val_info_2 = get_val_performance_data_2(test_scores, normal_scores, test_labels, topk=1)

        print('=========================** Result **============================\n')

        info = None
        if self.env_config['report'] == 'best':
            info = top1_best_info
        elif self.env_config['report'] == 'val':
            info = top1_val_info
        elif self.env_config['report'] == 'ad_best':
            info = top1_val_info_2

        print(f'F1 score: {info[0]}')
        print(f'precision: {info[1]}')
        print(f'recall: {info[2]}\n')
        return test_scores

    def get_save_path(self, feature_name=''):

        dir_path = self.env_config['save_path']

        if self.datestr is None:
            now = datetime.now()
            #            self.datestr = now.strftime('%m|%d-%H:%M:%S')
            self.datestr = now.strftime('%m_%d_%H_%M_%S')
        datestr = self.datestr

        paths = [
            #f'./pretrained/{dir_path}/best_{datestr}.pt',
            f'./pretrained2/{dir_path}/best_{datestr}.pt',
            f'./results/{dir_path}/{datestr}.csv',
        ]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-save_path_pattern', help='save path pattern', type=str, default='HAI')
    parser.add_argument('-dataset', help='wadi / swat/...', type=str, default='HAI')
    parser.add_argument('-device', help='cuda / cpu', type=str, default='cuda')
    parser.add_argument('-random_seed', help='random seed', type=int, default=2)
    parser.add_argument('-comment', help='experiment comment', type=str, default='')
    parser.add_argument('-report', help='best / val / ad_best', type=str, default='ad_best')
    parser.add_argument('-load_model_path', help='trained model path', type=str,default='')
    #parser.add_argument('-load_model_path', help='trained model path', type=str,default='./pretrained/PUMP/best_03_22_21_31_20.pt')
    #parser.add_argument('-load_model_path', help='trained model path', type=str, default='./pretrained2/sswat/best_03_24_19_10_07.pt')
    # optimization
    parser.add_argument('-batch', help='batch size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-2,
                        help='init learning rate')
    parser.add_argument('-epoch', help='train epoch', type=int, default=50)
    parser.add_argument('-decay', help='decay', type=float, default=0)
    parser.add_argument('-val_ratio', help='val ratio', type=float, default=0.2)
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--max_beta', type=float, default=1.0)
    parser.add_argument('--anneal_rate', type=float, default=0.05)

    # models
    parser.add_argument('-hidden_dim', help='hidden dimension', type=int, default=64)
    parser.add_argument('-slide_win', help='slide_win', type=int, default=56)
    parser.add_argument('-ts_patch', help='the patch size of MTS', type=int, default=8)
    parser.add_argument('-slide_stride', help='slide_stride of patch', type=int, default=8)
    parser.add_argument('-tsf_dim', help='feature dimension of ts', type=int, default=64)

    parser.add_argument('-use_stft', type=bool, default=False)
    parser.add_argument('-delay', help='delay embedding', type=int, default=3)
    parser.add_argument('-embedding', help='delay embedding', type=int, default=32)

    parser.add_argument('-num_query_token', help='number of query tokens', type=int, default=32)

    parser.add_argument('-img_size', help='image size', type=int, default=224)
    parser.add_argument('-patch_size', help='patch size of image', type=int, default=8)
    parser.add_argument('-embed_dim', type=int, default=256)
    parser.add_argument('-en_depth', help='depth of image encoder', type=int, default=6)
    parser.add_argument('-en_num_heads', help='heads of image encoder', type=int, default=4)

    parser.add_argument('-freeze_vit', type=bool, default=True)

    parser.add_argument('-out_layer_num', help='outlayer num', type=int, default=1)
    parser.add_argument('-out_layer_inter_dim', help='out_layer_inter_dim', type=int, default=256)

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)

    train_config = {
        'batch': args.batch,
        'seed': args.random_seed,
        'val_ratio':args.val_ratio,
        'epoch': args.epoch,
        'decay': args.decay,
        'hidden_dim': args.hidden_dim,
        'slide_win': args.slide_win,
        'ts_patch': args.ts_patch,
        'slide_stride': args.slide_stride,
        'tsf_dim': args.tsf_dim,
        'use_stft': args.use_stft,
        'delay': args.delay,
        'embedding': args.embedding,
        'img_size': args.img_size,
        'patch_size': args.patch_size,
        'embed_dim': args.embed_dim,
        'en_depth': args.en_depth,
        'en_num_heads': args.en_num_heads,
        'num_query_token': args.num_query_token,
        'freeze_vit': args.freeze_vit,
        'out_layer_num': args.out_layer_num,
        'out_layer_inter_dim': args.out_layer_inter_dim
    }

    env_config = {
        'save_path': args.save_path_pattern,
        'dataset': args.dataset,
        'report': args.report,
        'device': args.device,
        'load_model_path': args.load_model_path,
        'batch':args.batch
    }

    main = Main(train_config, env_config, debug=False)
    main.run()