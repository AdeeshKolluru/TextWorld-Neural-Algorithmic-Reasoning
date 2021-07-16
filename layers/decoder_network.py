import torch
import torch.nn as nn

class DecoderNetwork(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(DecoderNetwork, self).__init__()
        self.output_net = nn.Sequential(nn.Linear(in_features, out_features, bias=bias))

    def forward(self, x):
        dist = self.output_net(x)
        return dist

class DecoderNetwork_Maze(nn.Module):
    def __init__(self, in_features, bias=False):
        super(DecoderNetwork_Maze, self).__init__()
        self.output_net = nn.Sequential(nn.Linear(in_features, 1, bias=bias))

    def forward(self, x, edge_index):
        starting_edge_nodes_embs = torch.index_select(x, 0, edge_index[0])
        end_edge_nodes_embs = torch.index_select(x, 0, edge_index[1])
        cat_edge_node_embs = torch.cat((starting_edge_nodes_embs, end_edge_nodes_embs), 1)
        out = self.output_net(cat_edge_node_embs)
        return torch.sigmoid(out)
