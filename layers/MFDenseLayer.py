# -*- coding: utf-8 -*-
# @Time : 2022/2/22 15:08
# @Author : XDD
# @File : MFDenseLayer.py
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch import Tensor


class MFDenseLayer(nn.Module):
    """
    matrix factorization based dense layer
    reference:
    "https://github.com/panzheyi/MF-STN/blob/master/TaxiNYC/src/model/structure.py"
    "Matrix Factorization for Spatio-Temporal Neural Networks with Applications to Urban Flow Prediction"
    """
    def __init__(self, num_nodes, embed_dim, in_dim, out_dim, activation='Relu'):
        """
        :param num_nodes: 结点的数目
        :param embed_dim:  论文中 k
        :param in_dim: 输入每个点的特征维度， n_h
        :param out_dim: n_h_new
        :param activation: j激活函数的类型，Relu  or sigmoid
        """
        super(MFDenseLayer, self).__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.w1 = Parameter(Tensor(num_nodes, embed_dim), requires_grad=True)
        self.w2 = Parameter(Tensor(embed_dim, in_dim*out_dim), requires_grad=True)
        self.b1 = Parameter(Tensor(num_nodes, embed_dim), requires_grad=True)
        self.b2 = Parameter(Tensor(embed_dim, out_dim), requires_grad=True)
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.w1, gain=1)
        nn.init.xavier_normal_(self.w2, gain=1)
        # nn.init.kaiming_uniform_(self.w1, nonlinearity='relu')
        # nn.init.kaiming_uniform_(self.w2, nonlinearity='relu')
        nn.init.uniform_(self.b1)
        nn.init.uniform_(self.b2)

    def forward(self, data_in):
        """
        Forward process of MFDense layer
        :param data_in: tensor, shape: [b, n, in_dim]
        :return: tensor,shape:[b,n, out_dim]
        """
        weight = torch.mm(self.w1, self.w2).view(-1, self.in_dim, self.out_dim)  # [n, in_dim, out_dim]
        # weight = torch.reshape(torch.mm(self.w1, self.w2), (-1, self.in_dim, self.out_dim))
        bias = torch.mm(self.b1, self.b2).view(-1, self.out_dim)  # [n, 1, out_dim]
        # bias = torch.reshape(torch.mm(self.b1, self.b2), (-1, self.out_dim))
        out_put = torch.einsum('ijk,jkl ->ijl', data_in, weight) + bias
        # if self.activation == 'Relu':
        #     out_put = F.relu(out_put)
        # else:
        #     out_put = F.sigmoid(out_put)
        return out_put
