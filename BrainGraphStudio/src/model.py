import torch
from torch.nn import Linear, Sigmoid
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool,  GATConv, GATv2Conv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F

class TwoLayerSimpleGCN(torch.nn.Module):
    def __init__(self, num_features, embedding_size = 64):
        # Init parent
        super(TwoLayerSimpleGCN, self).__init__()
        torch.manual_seed(42)

        # GCN layers
        self.initial_conv = GCNConv(num_features, embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)

        # Output layer
        self.out = Linear(embedding_size*2, 1)

        self.sigmoid = Sigmoid()

    def forward(self, x, edge_index, edge_weight, batch_index):
        # First Conv layer
        hidden = self.initial_conv(x, edge_index, edge_weight=edge_weight)
        hidden = F.relu(hidden)

        # Other Conv layers
        hidden = self.conv1(hidden, edge_index, edge_weight=edge_weight)
        hidden = F.relu(hidden)
        hidden = self.conv2(hidden, edge_index, edge_weight=edge_weight)
        hidden = F.relu(hidden)

          
        # Global Pooling (stack different aggregations)
        hidden = torch.cat([gmp(hidden, batch_index), 
                            gap(hidden, batch_index)], dim=1)

        # Apply a final (linear) classifier.
        out = self.sigmoid(self.out(hidden))

        return out, hidden

class TwoLayerGCNWithPooling(torch.nn.Module):
    def __init__(self, indim, ratio, features):
        super(TwoLayerGCNWithPooling, self).__init__()
        torch.manual_seed(42)

        # GCN layers
        self.conv1 = GCNConv(indim, features)
        self.pool1 = TopKPooling(features, ratio=ratio, nonlinearity=torch.sigmoid)

        self.conv2 = GCNConv(features, features)
        self.pool2 = TopKPooling(features, ratio=ratio, nonlinearity=torch.sigmoid)
        
        self.conv3 = GCNConv(features, features)
        self.pool3 = TopKPooling(features, ratio=ratio, nonlinearity=torch.sigmoid)

        # Output layer
        self.out = Linear(features*2, 1)

        self.sigmoid = Sigmoid()

    def forward(self, x, edge_index, edge_weight, batch_index):
        # First Conv layer

        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x, edge_index, edge_weight, batch_index, _, _ = self.pool1(x, edge_index, edge_attr = edge_weight, batch = batch_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x, edge_index, edge_weight, batch_index, _, _ = self.pool2(x, edge_index, edge_attr = edge_weight, batch = batch_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index, edge_weight=edge_weight)
        x, edge_index, edge_weight, batch_index, _, _ = self.pool3(x, edge_index, edge_attr = edge_weight, batch = batch_index)
        x = F.relu(x)

        
        # Global Pooling (stack different aggregations)
        x = torch.cat([gmp(x, batch_index), 
                            gap(x, batch_index)], dim=1)

        # Apply a final (linear) classifier.
        out = self.sigmoid(self.out(x))

        return out, x


class TwoLayerAttGCNWithPooling(torch.nn.Module):
    def __init__(self, indim, ratio, features):
        super(TwoLayerAttGCNWithPooling, self).__init__()
        torch.manual_seed(42)

        # GCN layers
        self.conv1 = GATConv(indim, features)
        self.pool1 = TopKPooling(features, ratio=ratio, nonlinearity=torch.sigmoid)

        self.conv2 = GATConv(features, features)
        self.pool2 = TopKPooling(features, ratio=ratio, nonlinearity=torch.sigmoid)
        
        self.conv3 = GATConv(features, features)
        self.pool3 = TopKPooling(features, ratio=ratio, nonlinearity=torch.sigmoid)

        # Output layer
        self.out = Linear(features*2, 1)

        self.sigmoid = Sigmoid()

    def forward(self, x, edge_index, edge_weight, batch_index):
        # First Conv layer

        x = self.conv1(x, edge_index)
        x, edge_index, edge_weight, batch_index, _, _ = self.pool1(x, edge_index, batch = batch_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x, edge_index, edge_weight, batch_index, _, _ = self.pool2(x, edge_index, batch = batch_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x, edge_index, edge_weight, batch_index, _, _ = self.pool3(x, edge_index, batch = batch_index)
        x = F.relu(x)

        
        # Global Pooling (stack different aggregations)
        x = torch.cat([gmp(x, batch_index), 
                            gap(x, batch_index)], dim=1)

        # Apply a final (linear) classifier.
        out = self.sigmoid(self.out(x))

        return out, x


class TwoLayerAtt2GCNWithPooling(torch.nn.Module):
    def __init__(self, indim, ratio, features):
        super(TwoLayerAtt2GCNWithPooling, self).__init__()
        torch.manual_seed(42)

        # GCN layers
        self.conv1 = GATv2Conv(indim, features)
        self.pool1 = TopKPooling(features, ratio=ratio, nonlinearity=torch.sigmoid)

        self.conv2 = GATv2Conv(features, features)
        self.pool2 = TopKPooling(features, ratio=ratio, nonlinearity=torch.sigmoid)
        
        self.conv3 = GATv2Conv(features, features)
        self.pool3 = TopKPooling(features, ratio=ratio, nonlinearity=torch.sigmoid)

        # Output layer
        self.out = Linear(features*2, 1)

        self.sigmoid = Sigmoid()

    def forward(self, x, edge_index, edge_weight, batch_index):
        # First Conv layer

        x = self.conv1(x, edge_index)
        x, edge_index, edge_weight, batch_index, _, _ = self.pool1(x, edge_index, batch = batch_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x, edge_index, edge_weight, batch_index, _, _ = self.pool2(x, edge_index, batch = batch_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x, edge_index, edge_weight, batch_index, _, _ = self.pool3(x, edge_index, batch = batch_index)
        x = F.relu(x)

        
        # Global Pooling (stack different aggregations)
        x = torch.cat([gmp(x, batch_index), 
                            gap(x, batch_index)], dim=1)

        # Apply a final (linear) classifier.
        out = self.sigmoid(self.out(x))

        return out, x