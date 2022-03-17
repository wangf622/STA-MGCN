# -*- coding: utf-8 -*-
# @Time : 2021/12/20 16:05
# @Author : XDD
# @File : get_models.py
import sys


from models.STA_MGCN import STA_MGCN


import torch
import json
from layers.GCN import Adj_Preprocessor
from util import load_pickle, get_supports, get_sparse_supports, get_sparse_supports_cd
import argparse


def get_stamgcn(config_filename):
    with open(config_filename, 'r') as f:
        config = json.loads(f.read())

    data_config = config['Data']
    model_config = config['Model']

    device = model_config['device']
    # adj_filename = data_config['adj_filename']
    model_name = model_config['model_name']
    dataset_name = data_config['dataset_name']
    adj_matrix_name = data_config['adj_matrix_filename']
    num_of_nodes = int(data_config['num_of_nodes'])
    points_per_hour = int(data_config['points_per_hour'])
    num_for_predict = int(data_config['num_for_predict'])
    num_of_hours = data_config['num_of_hours']
    num_of_days = data_config['num_of_days']
    num_of_weeks = data_config['num_of_weeks']
    input_length = data_config['input_length']

    # model parameters
    kernel_type = model_config['kernel_type']
    in_channels = model_config['in_channels']
    K = model_config['K']
    gcn_filter = model_config['gcn_filter']  # 层数
    nb_st_block = model_config['nb_st_block']
    time_filter = model_config['time_filter']

    # adj
    if dataset_name in ["NYC_Flows"]:
        adjacent_adj, distance_adj, functionality_adj, od_adj, region_id_to_dict, _ = load_pickle(adj_matrix_name)
        adj_dis = torch.from_numpy(distance_adj).type(torch.FloatTensor)
        adj_func = torch.from_numpy(functionality_adj).type(torch.FloatTensor)
        adj_od = torch.from_numpy(od_adj).type(torch.FloatTensor)

        sta_adj_list = list()
        adj_preprocessor = Adj_Preprocessor(kernel_type=kernel_type, K=3)
        sta_adj_list.append(torch.unsqueeze(adj_preprocessor.process(adj_func), dim=0))
        sta_adj_list.append(torch.unsqueeze(adj_preprocessor.process(adj_od), dim=0))
        sta_adj_list.append(torch.unsqueeze(adj_preprocessor.process(adj_dis), dim=0))
        sta_adj_list = torch.cat(sta_adj_list, dim=0).to(device)
    elif dataset_name in ["Chengdu_Flows", "Chicago_Flows"]:
        adjacent_adj, distance_adj, od_adj, _ = load_pickle(adj_matrix_name)
        adj_dis = torch.from_numpy(distance_adj).type(torch.FloatTensor)
        adj_adjacent = torch.from_numpy(adjacent_adj).type(torch.FloatTensor)
        adj_od = torch.from_numpy(od_adj).type(torch.FloatTensor)

        sta_adj_list = list()
        adj_preprocessor = Adj_Preprocessor(kernel_type=kernel_type, K=3)
        sta_adj_list.append(torch.unsqueeze(adj_preprocessor.process(adj_adjacent), dim=0))
        sta_adj_list.append(torch.unsqueeze(adj_preprocessor.process(adj_od), dim=0))
        sta_adj_list.append(torch.unsqueeze(adj_preprocessor.process(adj_dis), dim=0))
        sta_adj_list = torch.cat(sta_adj_list, dim=0).to(device)
    else:
        raise SystemExit('wrong dataset!')

    # adj_list, _ = get_adj_list(adj_matrix_name)
    # adj_geo = adj_list[0].to(DEVICE)
    # adj_dis = adj_list[1].to(DEVICE)
    # adj_func = adj_list[2].to(DEVICE)
    # adj_od = adj_list[3].to(DEVICE)
    # 此时的邻接矩阵应该为A+I，应用 GCN 论文的 re_nor trick ,计算renormalize的邻接矩阵

    if model_name == "STA_MGCN":
        model = STA_MGCN(nb_st_block=nb_st_block, in_channels=in_channels, K=K, gcn_filter=gcn_filter, time_filter=time_filter,
                       num_for_predict=num_for_predict, num_nodes=num_of_nodes, num_weeks=num_of_weeks, num_days=num_of_days,
                       num_hours=num_of_hours, points_per_hour=points_per_hour, A=sta_adj_list)


    else:
        raise SystemExit('Wrong name of model!')

    return model




