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
from models.maze_network import MazeNetwork
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




def load_algorithms(algorithms, processor):
    hyperparameters = get_hyperparameters()
    DEVICE = hyperparameters["device"]
    DIM_LATENT = hyperparameters["dim_latent"]
    DIM_NODES_Maze = hyperparameters["dim_nodes_Maze"]
    DIM_EDGES = hyperparameters["dim_edges_Maze"]
    for algorithm in algorithms:
        if algorithm == "BellmanFord":
            # algo_net = models.BellFordNetwork(DIM_LATENT, DIM_NODES_BellmanFord, DIM_EDGES, processor, flow_datasets.BellmanFordDataset, './BellmanFord', use_ints=True, bits_size=8).to(DEVICE)
            algo_net = models.MazeNetwork(
                DIM_LATENT,
                DIM_NODES_Maze,
                DIM_EDGES,
                processor,
                MazeNetwork,
                "./MazeDataset",
            ).to(DEVICE)
        processor.add_algorithm(algo_net, algorithm)

    

def iterate_over(model, processor, optimizer, test=False):
    hyperparameters = get_hyperparameters()
    DEVICE = hyperparameters["device"]
    BATCH_SIZE = hyperparameters["maze_batch_size"]
    for algorithm in processor.algorithms.values():
        if processor.training:
            algorithm.iterator = iter(
                DataLoader(
                    algorithm.train_dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    drop_last=False,
                    num_workers=8,
                )
            )
        else:
            algorithm.iterator = iter(
                DataLoader(
                    algorithm.test_dataset if test else algorithm.val_dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    drop_last=False,
                    num_workers=8,
                )
            )
            algorithm.zero_validation_stats()
    try:
        while True:
            for algorithm in processor.algorithms.values():
                batch = next(algorithm.iterator)
                batch.to(DEVICE)
                EPS_I = 0
                with torch.set_grad_enabled(processor.training):
                    output = algorithm.process(batch, EPS_I)
                    if not processor.training:
                        algorithm.update_validation_stats(batch, output)

            if processor.training:
                processor.update_weights(optimizer)
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
    # NAME = (
    #         'BellmanFord'+args["--processor-type"]+str(hyperparameters["maze_lr"])+str(hyperparameters["maze_weight_decay"])
    #     )
    # print(torch.load(f'best_models/best_{NAME}.pt'))
    # processor.load_state_dict(torch.load(f'best_models/best_{NAME}.pt'))
    processor.load_processor_only(torch.load('best_models/processor_only.pt'))
    processor.algorithms["MazeNetwork"].load_termination_network(torch.load('best_models/termination_net.pt'))
    processor.eval()
    for param in processor.parameters():
        param.requires_grad = False

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
    
    maze_model = models.MazeNetwork(DIM_LATENT, OUT_DIM, DIM_NODE, DIM_EDGE, processor)


    with torch.autograd.profiler.profile(enabled=False, use_cuda=True) as prof:
        optimizer = optim.Adam(
            maze_model.parameters(),
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
