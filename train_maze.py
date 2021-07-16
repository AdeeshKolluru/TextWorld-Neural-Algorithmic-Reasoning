"""
Usage:
    train_bellmanford.py [options] [--algorithms=ALGO]...

Options:
    -h --help                        Show this screen.
    --algorithms ALGO                Which algorithms to add. One of {AugmentingPath, BFS, BellmanFord}. [default: BellmanFord]
    --model-name NAME                Specific name of model
    --processor-type PROC            Type of processor. One of {MPNN, PNA, GAT}. [default: MPNN]
"""
import copy
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric
from docopt import docopt
from torch_geometric.data import DataLoader

import flow_datasets
import models
from flow_datasets import MazeDataset, BellmanFordDataset
from hyperparameters import get_hyperparameters
from utils import interrupted, get_sizes_and_source



# def process(batch):
#     DEVICE = get_hyperparameters()["device"]
#     SIZE = batch.num_nodes
#     GRAPH_SIZES, SOURCE_NODES = get_sizes_and_source(batch)
#     x, y = get_input_output_features(batch, SOURCE_NODES)
#     return x, batch.edge_index, batch.edge_attr
    

def iterate_over(model, processor, optimizer):
    hyperparameters = get_hyperparameters()
    DEVICE = hyperparameters["device"]
    BATCH_SIZE = hyperparameters["maze_batch_size"]
    iterator = iter(
        DataLoader(
            processor.algorithms.values()[0].train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=False,
            num_workers=8,
        ))

    try:
        while True:
            batch = next(iterator)
            batch.to(DEVICE)
            with torch.set_grad_enabled(model.training):
                #node_fea, edge_index, edge_attr = process(batch)
                output = model(batch.x, batch.edge_index, batch.edge_attr, processor)
                loss = loss(output, True)
                loss.backward()
                optimizer.step()
            if interrupted():
                break
    except StopIteration:  # datasets should be the same size
        pass


if __name__ == "__main__":
    args = docopt(__doc__)
    hyperparameters = get_hyperparameters()
    DEVICE = hyperparameters["device"]
    DIM_LATENT = hyperparameters["dim_latent"]
    DIM_NODE = hyperparameters["dim_nodes_Maze"]
    DIM_EDGE = hyperparameters["dim_edges_Maze"]
    OUT_DIM = hyperparameters["dim_maze_out"]

    processor = models.AlgorithmProcessor(
        DIM_LATENT, None, args["--processor-type"]
    ).to(DEVICE)
    NAME = (
            'BellmanFord'+args["--processor-type"]+str(hyperparameters["maze_lr"])+str(hyperparameters["maze_weight_decay"])
        )
    # print(torch.load(f'best_models/best_{NAME}.pt'))
    processor.load_state_dict(torch.load(f'best_models/best_{NAME}.pt'))
    processor.eval()

    BATCH_SIZE = hyperparameters["batch_size"]
    PATIENCE_LIMIT = hyperparameters["patience_limit"]

    patience = 0
    # last_mean = 0
    # last_final = 0
    # last_loss = 0*1e9 if augmenting_path_network is not None else 1e9
    # cnt = 0
    best_final_acc = 0
    best_mean_acc = 0
    best_loss = np.inf

    # fmt = get_print_format()
    
    maze_model = models.MazeNetwork(DIM_LATENT, OUT_DIM, DIM_NODE, DIM_EDGE)


    with torch.autograd.profiler.profile(enabled=False, use_cuda=True) as prof:
        optimizer = optim.Adam(
            maze_model.params,
            lr=hyperparameters["lr"],
            weight_decay=hyperparameters["weight_decay"]
        )
        for epoch in range(3000):  # FIXME
            if interrupted():
                break
            # 0.0032
            maze_model.train()
            iterate_over(maze_model, processor, optimizer)

            patience += 1
            print("Epoch {:4d}: \n".format(epoch), end=" ")
            if patience >= PATIENCE_LIMIT:
                break
