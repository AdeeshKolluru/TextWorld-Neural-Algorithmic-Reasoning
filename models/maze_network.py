import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_cluster import random_walk

from hyperparameters import get_hyperparameters
from half_deterministic import obtain_paths, get_pairs
import utils
from layers import PredecessorNetwork, GAT
from models import AlgorithmBase
from overrides import overrides


class MazeNetwork(nn.Module):
    def __init__(self, latent_features, out_features, node_features, edge_features, algo_processor, bias=False):
        super(MazeNetwork, self).__init__()
        
        ne_input_features = node_features
        self.node_encoder = nn.Sequential(
            nn.Linear(ne_input_features, latent_features, bias=bias)
        )

        ee_input_features = edge_features
        self.edge_encoder = nn.Sequential(
            nn.Linear(ee_input_features, latent_features, bias=bias)
        )
        
        self.latent_decoder = nn.Sequential(nn.Linear(latent_features, out_features, bias=bias))

    def forward(self, node_input, edge_index, edge_attr, algo_processor):
        node_input = node_input.unsqueeze(1)
        encoded_nodes = self.node_encoder(node_input)
        #print(edge_attr.shape)
        #print(edge_attr)
        edge_attr = edge_attr.unsqueeze(dim=-1)
        #print(edge_attr.shape)
        encoded_edges = self.edge_encoder(edge_attr)

        latent_nodes = algo_processor(encoded_nodes, encoded_edges, utils.flip_edge_index(edge_index))

        output = self.latent_decoder(latent_nodes)
        return output

