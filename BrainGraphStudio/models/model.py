from BrainGraphStudio.BrainGB.src.models import GAT, GCN, BrainNN, MLP
from BrainGraphStudio.models.brainGNN.braingnn import Network
import torch
from .params import Param


def build_model(args, device, model_name, num_features, num_nodes, n_MLP_layers, hidden_dim=None, n_classes = 2):
    print(model_name)
    if model_name == 'gcn':
        model = BrainNN(args,
                      GCN(num_features, args, num_nodes, n_classes),
                      MLP(2 * num_nodes, hidden_dim, n_MLP_layers, torch.nn.ReLU, n_classes=n_classes),
                      ).to(device)
    elif model_name == 'gat':
        model = BrainNN(args,
                       GAT(num_features, args, num_nodes, n_classes),
                      MLP(2 * num_nodes, hidden_dim, n_MLP_layers, torch.nn.ReLU, n_classes=n_classes),
                      ).to(device)
    elif model_name == "brainGNN":
        model = Network(num_features, args.pooling_ratio, n_classes, R = num_features, k = args.communities)
    else:
        raise ValueError(f"ERROR: Model variant \"{model_name}\" not found!")
    return model


class BrainGNN():
    dataparams = [Param("batchsize", 32, int), Param("train_split", 0.60, float), 
                  Param("validation_split", 0.2, float), Param("test_split", 0.2, float),
                  Param("random_seed", 0, int, description = "Random Seed for All Random Operations. Leave at 0 to Disable")]
    trainparams = [Param("lr", 0.01, float, optimizable=True, default_search_space=[0.001, 0.0005, 1]),
                Param("epochs", 100, int, optimizable=True, default_search_space=[50,80,100]), 
                Param("k_fold_splits", 0, int),
              Param("weight_decay", 5e-3, float, optimizable=True), 
              Param("gamma", 0.5, float, optimizable = True), 
              Param("lr_scheduler_stepsize", 20, int, optimizable = True, description="decay the learning rate by gamma every <lr_sceduler_stepsize> epochs"),
                Param("lamb0", 1, float, optimizable=True), 
                Param("lamb1", 0, float, optimizable=True),
                Param("lamb2", 0, float, optimizable=True), 
                Param("lamb3", 0.1, float, optimizable=True),
                Param("lamb4", 0.1, float, optimizable=True), 
                Param("lamb5", 0.1, float, optimizable=True),
                Param("communities", 3, int, optimizable=True, default_search_space=[3,7,10]),
                Param("test_interval", 5, int)
                ]
    
    architecture_params = [Param("n_GNN_layers", 2, int, optimizable=True), 
                           Param("pooling_ratio", 0.5, float, optimizable=True)]
    params = {"data":dataparams, "train":trainparams, "architecture": architecture_params}


class BrainGB():
    dataparams = [Param("batchsize", 32, int), Param("train_split", 0.60, float), 
                  Param("validation_split", 0.2, float), Param("test_split", 0.2, float),
                  Param("random_seed", 0, int, description = "Random Seed for All Random Operations. Leave at 0 to Disable")]
    trainparams = [Param("lr", 0.01, float, optimizable=True, default_search_space=[0.1, 0.01, 0.001, 0.0001]), 
                   Param("epochs", 100, int, optimizable=True, default_search_space=[50,80,100]),
                   Param("weight_decay", 1e-4, float, optimizable=True, default_search_space= [1e-5, 1e-4, 1e-3]),
                   Param("k_fold_splits", 0, int), Param("dropout", 0.5, float),
                   Param("test_interval", 5, int), Param('bucket_sz', 0.05, float), Param("num_heads", 2, int)]
    architecture_params = [Param("n_GNN_layers", 2, int, optimizable=True, default_search_space=[1, 2, 3, 4]), 
                           Param("n_MLP_layers", 1, int, optimizable=True, default_search_space=[1, 2, 3, 4]), 
                           Param("hidden_dim", 256, int, optimizable=True, default_search_space=[8, 12, 16, 32]), 
                           Param("edge_emb_dim", 256, int, optimizable=True, default_search_space=[32, 64, 96, 128, 256, 512, 1024])]
    params = {"data":dataparams, "train":trainparams, "architecture": architecture_params}