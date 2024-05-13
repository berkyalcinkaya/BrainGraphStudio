import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset, Data
from torch import tensor, from_numpy
import numpy as np
from scipy.sparse import coo_matrix
from BrainGraphStudio.BrainGB.src.dataset.transforms import *
from BrainGraphStudio.BrainGB.src.dataset.brain_data import BrainData
import random
import tqdm
import logging

logger = logging.getLogger(__name__)

def is2D(np_array):
    return len(np_array.shape)==2

def density_threshold(networks, density_percentile = 10, v = False, positive_only = False):
    if is2D(networks):
        networks = np.expand_dims(networks, 0)
    new_networks = networks.copy()
    for i in tqdm(range(networks.shape[0])):
        if positive_only:
            thresh = np.percentile(networks[i][networks[i]>=0], density_percentile)
        else:
            thresh = np.percentile(networks[i], density_percentile)
        if v:
            print("Threshold for", i, ": ", thresh)
        new_networks[i,:,:][new_networks[i]<thresh] = 0
    
    if is2D(networks):
        return new_networks[0,:,:]
    return new_networks 

def expand_list(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

def get_train_test_split_grouped(datas: list[Data], groupby = 15, split = 0.80):
    if len(datas) % groupby != 0:
        raise ValueError("Given list must be  a multiple of split param")
    grouped_list = [datas[i:i + groupby] for i in range(0, len(datas), groupby)]
    train, test = get_train_test_split(grouped_list, split = split)
    return expand_list(train), expand_list(test)

def get_train_test_split(datas: list[Data], split = 0.80, choose_random = False, seed = None):
    if seed is not None:
        random.seed(seed)
    data_len = len(datas)
    num_train = int(split*data_len)
    idxs = list(range(data_len))
    train_idxs = random.sample(idxs, num_train)
    train = []
    test = []
    for i in idxs:
        if i in train_idxs:
            train.append(datas[i])
        else:
            test.append(datas[i])
    return train, test


def get_transform(transform_type: str):
    """
    Maps transform_type to transform class
    :param transform_type: str
    :return: BaseTransform
    """
    if transform_type == 'identity':
        return Identity()
    elif transform_type == 'degree':
        return Degree()
    elif transform_type == 'degree_bin':
        return DegreeBin()
    elif transform_type == 'LDP':
        return LDPTransform()
    elif transform_type == 'adj':
        return Adj()
    elif transform_type == 'node2vec':
        return Node2Vec()
    elif transform_type == 'eigenvector':
        return Eigenvector()
    elif transform_type == 'eigen_norm':
        return EigenNorm()
    else:
        raise ValueError('Unknown transform type: {}'.format(transform_type))

def apply_transforms(datas, transform_str):
    transform = get_transform(transform_str)
    datas = [transform(data) for data in datas]
    return datas

def adjacency_matrix_to_coo(adjacency_matrix):
    coo = coo_matrix(adjacency_matrix)
    return coo.data, np.array([coo.row, coo.col], dtype = np.uint8)

def convert_raw_to_datas(X, Y):
    logger.info(str(X.shape))
    logger.info(str(Y.shape))
    datas = []
    for i in range(X.shape[0]):
        subnet = X[i]
        edge_weights, coo_format = adjacency_matrix_to_coo(subnet)
        y_i = tensor(np.array([Y[i]])).to(torch.int64)
        datas.append(Data(x=from_numpy(subnet).to(torch.float32), edge_index=from_numpy(coo_format).to(torch.int64), 
                    edge_attr = from_numpy(edge_weights).unsqueeze(1).to(torch.float32), y=y_i,
                    pos = from_numpy(np.identity(subnet.shape[-1])).to(torch.float32)))
        datas[-1].num_nodes = subnet.shape[0]
        
    return datas
