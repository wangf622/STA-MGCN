# -*- coding: utf-8 -*-
# @Time : 2021/12/20 16:26
# @Author : XDD
# @File : ASTGCN.py
import torch
import torch.nn as nn
from layers.GCN import GCN
import torch.nn.functional as F
from layers.CausalConv2d import CausalConv2d
from torch.nn import Conv2d
from torch import Tensor
from layers.MFDenseLayer import MFDenseLayer


class MultiGraphConv(nn.Module):
    """
    多图卷积
    """
    def __init__(self, K, input_dim: int, out_dim: int, agg='sum'):
        """

        :param K: 用于区分不同的图卷积操作，GCN 为1；chebconv=3
        :param input_dim: 输入
        :param out_dim:
        :param agg : 多图的卷积的聚合方式，默认为sum
        """
        super().__init__()
        self.agg = agg
        self.gcn1 = GCN(K, input_dim, out_dim, bias=True, activation=nn.ReLU)
        self.gcn2 = GCN(K, input_dim, out_dim, bias=True, activation=nn.ReLU)
        self.gcn3 = GCN(K, input_dim, out_dim, bias=True, activation=nn.ReLU)

    def reset_parameters(self):
        self.gcn1.reset_parameters()
        self.gcn2.reset_parameters()
        self.gcn3.reset_parameters()

    def forward(self, x, A):
        """
        :param A: A: support adj matrices - torch.Tensor (3, K, n_nodes, n_nodes)
        :param x: graph feature/signal - torch.Tensor (batch_size, n_nodes, input_dim)
        :return:
        """
        out1 = self.gcn1(A[0, :, :, ], x)
        out2 = self.gcn2(A[1, :, :, ], x)
        out3 = self.gcn3(A[2, :, :, ], x)
        if self.agg == 'sum':
            return out1+out2+out3
        elif self.agg == 'max':
            out = [torch.unsqueeze(out1, dim=0), torch.unsqueeze(out1, dim=0), torch.unsqueeze(out1, dim=0)]
            out = torch.cat(out, dim=0)
            return torch.max(out, dim=0)
        else:
            raise ValueError(f'ERROR: activation function {self.agg} is not defined.')


class TemporalAttentionLayer(nn.Module):
    """
    计算时间注意力
    """
    def __init__(self, in_channels, num_nodes, num_time_steps):
        super(TemporalAttentionLayer, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(num_nodes))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_nodes))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels))
        self.be = nn.Parameter(torch.FloatTensor(1, num_time_steps, num_time_steps))
        self.Ve = nn.Parameter(torch.FloatTensor(num_time_steps, num_time_steps))
        # self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.U1)
        nn.init.xavier_normal_(self.U2, gain=1)
        nn.init.uniform_(self.U3)
        nn.init.xavier_normal_(self.be, gain=1)
        nn.init.xavier_normal_(self.Ve, gain=1)

    def forward(self, x):
        """
        “:param x (batch_size, num_nodes, f_in, T) b,n,f,t
        :return: (B, T ,T)加权系数
        """
        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        # x:(B, N, F_in, T) -> (B, T, F_in, N)
        # (B, T, F_in, N)(N) -> (B,T,F_in)
        # (B,T,F_in)(F_in,N)->(B,T,N)

        rhs = torch.matmul(self.U3, x)  # (F)(B, N, F, T) -> (B, N, T)
        product = torch.matmul(lhs, rhs)  # (B,T,N)(B,N,T)->(B,T,T)
        e = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)
        e_normalized = F.softmax(e, dim=1)  # 经过softmax函数，计算得到 时间维度上的注意力系数
        return e_normalized


class STA_MGCN_block(nn.Module):
    """
    时间attention  mask*x; 之后多图卷积，AXW形式；之后时间卷积
    """
    def __init__(self, K, input_dim, gcn_filter, num_nodes, time_filter, num_time_steps, time_stride=1):
        """

        :param K: 图卷积的层数
        :param input_dim: 输入的特征维度
        :param gcn_filter: 图卷积输出的特征的维度
        :param num_nodes: 节点的个数
        :param time_filter: 时间轴上卷积的输出的维度
        :param time_strides: 时间轴上卷积的步进长度
        :param num_time_steps: 一共的输入的点数
        """
        super(STA_MGCN_block, self).__init__()
        self.t_att = TemporalAttentionLayer(in_channels=input_dim, num_nodes=num_nodes, num_time_steps=num_time_steps)
        self.multi_gcn = MultiGraphConv(K=K, input_dim=input_dim, out_dim=gcn_filter, agg='sum')
        self.time_conv = Conv2d(gcn_filter, 2 * time_filter, kernel_size=(1, 3), padding=(0, 1),
                                stride=(1, 1), bias=True, dilation=2)
        self.residual_conv = nn.Conv2d(in_channels=input_dim, out_channels=time_filter, kernel_size=(1, 1), stride=(1, time_stride))
        self.ln = nn.LayerNorm(time_filter)
        self.time_filter = time_filter

    def reset_parameters(self):
        self.t_att.reset_parameters()
        self.multi_gcn.reset_parameters()

    def forward(self, x, A):
        """

        :param x: (B, N, F, T)
        :return: (B, N, time_filter, T)
        """
        batch_size, num_nodes, num_features, num_time_steps = x.shape
        t_att = self.t_att(x)
        x_t_att = torch.matmul(x.reshape(batch_size, -1, num_time_steps), t_att).\
            reshape(batch_size, num_nodes, num_features, num_time_steps)

        spatial_gcn = []
        for t in range(num_time_steps):
            graph_signal = x_t_att[:, :, :, t]
            output = self.multi_gcn(graph_signal, A)
            spatial_gcn.append(torch.unsqueeze(output, dim=-1))
        spatial_gcn_out = torch.cat(spatial_gcn, dim=-1)

        t_x = self.time_conv(spatial_gcn_out.permute(0, 2, 1, 3))
        t_x1, t_x2 = torch.split(t_x, [self.time_filter, self.time_filter], dim=1)
        time_conv_out = torch.tanh(t_x1) * torch.sigmoid(t_x2)

        # (b,N,F,T)->(b,F,N,T)
        # residual shortcut
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))  # (b,N,F,T)->(b,F,N,T)
        x_residual = self.ln(F.relu(x_residual + time_conv_out).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)
        # (b,F,N,T)->(b,T,N,F) -ln-> (b,T,N,F)->(b,N,F,T)
        return x_residual


class STA_MGCN_submodule(nn.Module):
    def __init__(self, nb_st_block, in_channels, K, gcn_filter, time_filter, num_for_predict, num_time_steps, num_nodes):
        """

        :param nb_st_block: 时空块个数
        :param in_channels:
        :param K: 图卷积的次数
        :param gcn_filter: 图卷积输出 dim
        :param time_filter: 64
        :param num_for_predict:2
        :param num_time_steps: T
        :param num_nodes: 63
        """
        super(STA_MGCN_submodule, self).__init__()
        self.BlockList = nn.ModuleList([STA_MGCN_block(K=K, input_dim=in_channels, gcn_filter=gcn_filter, time_filter=time_filter,
                                                     num_time_steps=num_time_steps, num_nodes=num_nodes)])
        self.BlockList.extend([STA_MGCN_block(K=K, input_dim=time_filter, gcn_filter=gcn_filter,
                                            time_filter=time_filter, num_time_steps=num_time_steps, num_nodes=num_nodes)
                               for _ in range(nb_st_block-1)])

        self.predict_length = num_for_predict
        self.time_filter = time_filter

        # self.fc_2 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Linear(num_time_steps*time_filter, out_features=time_filter),
        #     nn.ReLU(),
        #     nn.Linear(in_features=time_filter, out_features=2)
        # )
        self.fc_3 = nn.Sequential(
            nn.ReLU(),
            MFDenseLayer(num_nodes, embed_dim=4, in_dim=num_time_steps * time_filter, out_dim=32),
            nn.ReLU(),
            MFDenseLayer(num_nodes, embed_dim=4, in_dim=32, out_dim=2)
        )

    def reset_parameters(self):
        for block in self.BlockList:
            block.reset_parameters()
        # self.fc_in.reset_parameters()
        # self.fc_out.reset_parameters()

    def forward(self, x, A):
        """

        :param x: 数据输入，B,N,F,T_h
        :param A: adj_list, (3, N, N) tensor
        :return:
        """

        for block in self.BlockList:
            x = block(x, A)

        # return x
        batch_size, num_nodes, feature_dim, num_time_slot = x.shape
        # INFLOW  OUTFLOW
        x = x.reshape(-1, num_nodes, feature_dim*num_time_slot)
        out = self.fc_3(x)
        return out  # B N 2


class STA_MGCN(nn.Module):
    def __init__(self, nb_st_block, in_channels, K, gcn_filter, time_filter, num_for_predict, num_nodes,
                 num_hours, num_weeks, num_days, points_per_hour, A):
        """

        :param nb_st_block: 时空块的个数
        :param in_channels: 输入
        :param k: 图卷积的阶数
        :param gcn_filter: 图卷积的输出
        :param time_filter: 时间卷积的额输出
        :param num_for_predict:
        :param num_nodes:
        :param num_hours:
        :param num_weeks:
        :param num_days:
        :param points_per_hour:
        :param A:
        """
        super(STA_MGCN, self).__init__()
        self.predict_length = num_for_predict
        self.num_nodes = num_nodes
        self.Th = num_hours
        # self.Td = num_days*points_per_hour//2
        self.Td = num_days
        self.Tw = num_weeks

        self.STA_MGCN_h = STA_MGCN_submodule(nb_st_block=nb_st_block, in_channels=in_channels, K=K, gcn_filter=gcn_filter, time_filter=time_filter,
                                         num_time_steps=num_hours, num_nodes=num_nodes, num_for_predict=num_for_predict)
        self.STA_MGCN_d = STA_MGCN_submodule(nb_st_block=nb_st_block, in_channels=in_channels, K=K, gcn_filter=gcn_filter,
                                         time_filter=time_filter,
                                         num_time_steps=self.Td, num_nodes=num_nodes,
                                         num_for_predict=num_for_predict)
        self.STA_MGCN_w = STA_MGCN_submodule(nb_st_block=nb_st_block, in_channels=in_channels, K=K, gcn_filter=gcn_filter,
                                         time_filter=time_filter,
                                         num_time_steps=3, num_nodes=num_nodes,
                                         num_for_predict=num_for_predict)
        self.W_h = nn.Parameter(torch.FloatTensor(num_nodes, 2))
        self.W_d = nn.Parameter(torch.FloatTensor(num_nodes, 2))
        self.W_w = nn.Parameter(torch.FloatTensor(num_nodes, 2))

        self.adj = A

    def reset_parameters(self):
        self.astgcn_h.reset_parameters()
        self.astgcn_d.reset_parameters()
        self.astgcn_w.reset_parameters()

        # self.fc1.reset_parameters()
        # self.fc2.reset_parameters()
        nn.init.xavier_normal_(self.W_d, gain=1)
        # nn.init.kaiming_uniform_(self.W_d, nonlinearity='relu')
        # nn.init.kaiming_uniform_(self.W_h, nonlinearity='relu')
        # nn.init.kaiming_uniform_(self.W_w, nonlinearity='relu')
        #
        nn.init.xavier_normal_(self.W_h, gain=1)
        nn.init.xavier_normal_(self.W_w, gain=1)

    def forward(self, x):
        """
        B F N T -> B N F T
        :param x:
        :return:
        """
        x = x.permute(0, 2, 1, 3)
        x_tw = x[:, :, :, :self.Tw]
        # 生成0值tensor
        if self.Tw < 3:
            x_tw = nn.functional.pad(x_tw, (3 - self.Tw, 0, 0, 0))

        x_td = x[:, :, :, self.Tw:self.Tw + self.Td]
        if self.Td < 3:
            x_tw = nn.functional.pad(x_td, (3 - self.Td, 0, 0, 0))

        x_th = x[:, :, :, self.Tw+self.Td:]
        out_th = self.STA_MGCN_h(x_th, self.adj)
        # b, n, f, t_h = out_th.shape
        # out_th = torch.reshape(input=out_th, shape=(-1, n, f*t_h))
        out_td = self.STA_MGCN_d(x_td, self.adj)
        # _, _, _, t_d = out_td.shape
        # out_td = torch.reshape(input=out_td, shape=(-1, n, f*t_d))
        out_tw = self.STA_MGCN_w(x_tw, self.adj)

        out = torch.mul(out_th, self.W_h) + torch.mul(out_tw, self.W_w) + torch.mul(out_td, self.W_d)  # b n 2

        return torch.relu(out)


