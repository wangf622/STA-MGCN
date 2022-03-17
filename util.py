import pickle
import numpy as np
import os
import torch
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from torch_sparse import SparseTensor
from scipy import sparse


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, begin=0, days=48, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size  # num_padding padding的次数
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)  # 重复 最后一个值， num_padding次,得到一个list
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)  # 拼接复制的list和原来的数据
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        self.ind = np.arange(begin, begin + self.size)
        self.days = days

    def shuffle(self):
        permutation = np.random.permutation(self.size)  # 随机生成一个长度为数据长度的数组
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.ind = self.ind[permutation]   # ndarray ,打乱索引值
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                # current_ind 为当前的batch的序列值
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                i_i = (self.ind[start_ind: end_ind, ...] % self.days)
                # xi_i = np.tile(np.arange(x_i.shape[1]), [x_i.shape[0], x_i.shape[2], 1, 1]).transpose(
                #     [0, 3, 1, 2]) + self.ind[start_ind: end_ind, ...].reshape([-1, 1, 1, 1])
                # x_i = np.concatenate([x_i, xi_i % self.days / self.days, np.eye(7)[xi_i // self.days % 7].squeeze(-2)],
                #                      axis=-1)
                yield x_i, y_i, i_i
                self.current_ind += 1

        return _wrapper()


class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None, days=24, out_seq=1,
                 in_seq=10):
    data = {}
    device = torch.device('cuda:0')
    print('load data from:%s' % dataset_dir)
    for category in ['train', 'val', 'test']:
        cat_data = np.load(dataset_dir, allow_pickle=True)
        data['x_' + category] = cat_data['x_'+category][:, -in_seq:, :, :]  # B T N F (inflow, outflow)
        data['y_' + category] = cat_data['y_'+category][:, :out_seq, :, :]

        if category == "train":
            data['scaler'] = StandardScaler(mean=cat_data['x_'+category].mean(axis=(0, 1, 2), keepdims=True), std=cat_data['x_'+category].std(axis=(0, 1, 2), keepdims=True))
            # [...,0] 表示最后一个维度数据
    # 归一化处理，只对输入进行处理
    for category in ['train', 'val', 'test']:
        data['x_' + category] = data['scaler'].transform(data['x_' + category])  # 只对input做处理

    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, days=days, begin=0)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size, days=days,
                                    begin=data['x_train'].shape[0])
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size, days=days,
                                     begin=data['x_train'].shape[0] + data['x_val'].shape[0])
    return data


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val))


def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)  # mask ： 不为nan的地方 为bool True
        else:
            mask = np.not_equal(labels, null_val)  # != null_val 的地方True
        mask = mask.astype('float32')  # 转换数据类型为 float
        mask /= np.mean(mask)
        mse = np.square(np.subtract(preds, labels)).astype('float32')
        mse = np.nan_to_num(mse * mask)
        return np.mean(mse)


def masked_mape_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels))
        mape = np.nan_to_num(mask * mape)
        return 100 * np.mean(mape)


def masked_mae_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse


def metric_np(pred, real):
    mae = masked_mae_np(pred, real, 0)
    mape = masked_mape_np(pred, real, 0)
    rmse = masked_rmse_np(pred, real, 0)
    return mae, mape, rmse


def get_supports(adj_mx, adj_type, device):
    # 根据邻接矩阵的类型的不同，进行不同的normalization
    if adj_type == "scaled_laplacian":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adj_type == "normalized_laplacian":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adj_type == "sym_adj":
        adj = [sym_adj(adj_mx)]
    elif adj_type == "transition":
        adj = [asym_adj(adj_mx)]
    elif adj_type == "double_transition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adj_type == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    supports = [torch.tensor(i).to(device) for i in adj]
    return supports


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


def get_sparse_adj(adj, edge_vec):
    num_nodes = adj.shape[0]
    adj_sparse = sparse.coo_matrix(adj)
    row = torch.tensor(adj_sparse.row, dtype=torch.long)  # 行
    col = torch.tensor(adj_sparse.col, dtype=torch.long)  # 列
    new_edge_vec = np.expand_dims(adj, axis=-1) * edge_vec  # 先做element-wise 乘法
    new_edge_vec_sparse_0 = sparse.coo_matrix(new_edge_vec[:, :, 0]).data  # edge_vec 1
    new_edge_vec_sparse_1 = sparse.coo_matrix(new_edge_vec[:, :, 1]).data  # edge_vec 2

    # 得到边的权重向量
    value = np.concatenate((np.expand_dims(new_edge_vec_sparse_0, axis=-1), np.expand_dims(new_edge_vec_sparse_1, axis=-1)), axis=-1)
    # %%
    value = torch.tensor(data=value, dtype=torch.float32)
    adj_processed = SparseTensor(row=row, col=col, value=value, sparse_sizes=(num_nodes, num_nodes))  # 使用稀疏张量的形式表示邻接矩阵
    return adj_processed


def get_sparse_supports(adj_file_name):
    adj_geo, adj_dis, adj_func, adj_od, _, edge_vec = load_pickle(adj_file_name)
    supports = []
    adj_geo_sparse = get_sparse_adj(adj_geo, edge_vec=edge_vec)
    supports.append(adj_geo_sparse)
    adj_dis_sparse = get_sparse_adj(adj_dis, edge_vec=edge_vec)
    supports.append(adj_dis_sparse)
    adj_func_sparse = get_sparse_adj(adj_func, edge_vec=edge_vec)
    supports.append(adj_func_sparse)
    adj_od_sparse = get_sparse_adj(adj_od, edge_vec=edge_vec)
    supports.append(adj_od_sparse)
    return supports


def get_sparse_supports_cd(adj_file_name):
    adj_geo, adj_dis, adj_od, edge_vec = load_pickle(adj_file_name)
    supports = []
    adj_geo_sparse = get_sparse_adj(adj_geo, edge_vec=edge_vec)
    supports.append(adj_geo_sparse)
    adj_dis_sparse = get_sparse_adj(adj_dis, edge_vec=edge_vec)
    supports.append(adj_dis_sparse)
    adj_od_sparse = get_sparse_adj(adj_od, edge_vec=edge_vec)
    supports.append(adj_od_sparse)
    return supports
