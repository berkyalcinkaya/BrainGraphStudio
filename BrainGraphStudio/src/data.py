from numpy import dtype
import numpy as np
from .utils import copy_elements
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from torch import tensor, from_numpy
import random
from .utils import *

def get_labels_augmented(path, aug_size = 15):
    y = pd.read_csv(path).iloc[:,-1].to_list()
    return np.array(copy_elements(y, aug_size), dtype=np.uint8)

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

def get_data_NCANDA(subnetwork = "DMN", batch_size = 32, train_percent = 0.80):
    y = pd.read_csv("ID_Drink_QC1_FY3.csv").iloc[:,-1].to_numpy().astype(np.uint8)
    networks = load_mat_flist("siteE_correlation_weighted_191subs.flist")
    networks_thresh = density_threshold(networks)
    subnetworks = extract_subnetcrworks_from_csv("ShenAtlas_Subnetworks.csv", networks_thresh)[subnetwork]
    datas = []
    for i in range(subnetworks.shape[0]):
        subnet = subnetworks[i]
        edge_weights, coo_format = adjacency_matrix_to_coo(subnet)
        y_i = tensor(np.array([y[i]])).unsqueeze(1)
        datas.append(Data(x=from_numpy(subnet), edge_index=from_numpy(coo_format).to(torch.int64), edge_weights = from_numpy(edge_weights), y=y_i))

    num_examples = len(datas)
    train_datas, test_datas = get_train_test_split(datas, split = train_percent, seed =1)
    print(f"num training examples: {len(train_datas)}, num validation examples: {len(test_datas)}, total: {num_examples}")
    
    train_loader = DataLoader(train_datas, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(test_datas, batch_size = len(test_datas), shuffle = False)
    return datas, train_loader, val_loader

def get_data(X,Y, batch_size = 32, train_percent = 0.80):
    datas = []
    for i in range(X.shape[0]):
        subnet = X[i]
        edge_weights, coo_format = adjacency_matrix_to_coo(subnet)
        y_i = tensor(np.array([Y[i]])).to(torch.int64)
        datas.append(Data(x=from_numpy(subnet).to(torch.float32), edge_index=from_numpy(coo_format).to(torch.int64), 
                    edge_attr = from_numpy(edge_weights).unsqueeze(1).to(torch.float32), y=y_i,
                    pos = from_numpy(np.identity(subnet.shape[-1])).to(torch.float32)))
    num_examples = len(datas)
    train_datas, test_datas = get_train_test_split(datas, split = train_percent)
    
    #print(f"num training examples: {len(train_datas)}, num validation examples: {len(test_datas)}, total: {num_examples}")
    
    train_loader = DataLoader(train_datas, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(test_datas, batch_size = len(test_datas), shuffle = False)
    return datas, train_loader, val_loader


def get_data_grouped(X, Y, batch_size = 32, train_percent = 0.80):
    datas = []
    for i in range(X.shape[0]):
        subnet = X[i]
        edge_weights, coo_format = adjacency_matrix_to_coo(subnet)
        y_i = tensor(np.array([Y[i]])).to(torch.int64)
        datas.append(Data(x=from_numpy(subnet).to(torch.float32), edge_index=from_numpy(coo_format).to(torch.int64), 
                    edge_attr = from_numpy(edge_weights).unsqueeze(1).to(torch.float32), y=y_i,
                    pos = from_numpy(np.identity(subnet.shape[-1])).to(torch.float32)))

    num_examples = len(datas)
    train_datas, test_datas = get_train_test_split_grouped(datas, split = train_percent)
    print(f"num training examples: {len(train_datas)}, num validation examples: {len(test_datas)}, total: {num_examples}")
    
    train_loader = DataLoader(train_datas, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(test_datas, batch_size = len(test_datas), shuffle = True)
    return datas, train_loader, val_loader