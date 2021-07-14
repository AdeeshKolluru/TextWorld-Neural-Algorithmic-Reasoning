"""
Usage:
    train.py [options] [--algorithms=ALGO]...

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
from flow_datasets import SingleIterationDataset, BellmanFordDataset
from hyperparameters import get_hyperparameters
from utils import interrupted


def get_print_format():
    fmt = """
==========================
Mean step acc: {:.4f}    Last step acc: {:.4f}
loss-(dist,pred,term,total): {:.4f} {:.4f} {:.4f} {:.4f}
patience: {}
===============
"""
    return fmt


def iterate_over(processor, optimizer=None, test=False):
    hyperparameters = get_hyperparameters()
    DEVICE = hyperparameters["device"]
    BATCH_SIZE = hyperparameters["batch_size"]

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
                start = time.time()
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


def load_algorithms(algorithms, processor, use_ints):
    hyperparameters = get_hyperparameters()
    DEVICE = hyperparameters["device"]
    DIM_LATENT = hyperparameters["dim_latent"]
    DIM_NODES_BellmanFord = hyperparameters["dim_nodes_BellmanFord"]
    DIM_EDGES = hyperparameters["dim_edges_BellmanFord"]
    for algorithm in algorithms:
        if algorithm == "BellmanFord":
            # algo_net = models.BellFordNetwork(DIM_LATENT, DIM_NODES_BellmanFord, DIM_EDGES, processor, flow_datasets.BellmanFordDataset, './BellmanFord', use_ints=True, bits_size=8).to(DEVICE)
            algo_net = models.BellFordNetwork(
                DIM_LATENT,
                DIM_NODES_BellmanFord,
                DIM_EDGES,
                processor,
                BellmanFordDataset,
                "./BellmanFord",
            ).to(DEVICE)
        processor.add_algorithm(algo_net, algorithm)


def get_print_info(bf_network):
    (
        total_loss_dist,
        total_loss_pred,
        total_loss_term,
    ) = bf_network.get_validation_losses()
    mean_step, final_step = bf_network.get_validation_accuracies()
    total_loss = total_loss_dist + total_loss_pred + total_loss_term
    # broken_invariants, broken_reachabilities, broken_flows, broken_all = bf_network.get_broken_invariants()
    # len_broken = len(broken_invariants)
    return (
        total_loss_dist,
        total_loss_pred,
        total_loss_term,
        total_loss,
        mean_step,
        final_step,
    )  # , tnr, subtract_acc, broken_invariants, broken_reachabilities, broken_flows, broken_all, len_broken


if __name__ == "__main__":

    args = docopt(__doc__)
    hyperparameters = get_hyperparameters()
    DEVICE = hyperparameters["device"]
    DIM_LATENT = hyperparameters["dim_latent"]
    DIM_EDGES = hyperparameters["dim_edges"]
    NAME = (
        args["--model-name"]
        if args["--model-name"] is not None
        else datetime.now().strftime("%b-%d-%Y-%H-%M")
    )

    processor = models.AlgorithmProcessor(
        DIM_LATENT, SingleIterationDataset, args["--processor-type"]
    ).to(DEVICE)
    print("PARAMETERS", sum(p.numel() for p in processor.parameters()))
    print(list((name, p.numel()) for name, p in processor.named_parameters()))
    load_algorithms(args["--algorithms"], processor, True)
    # processor.reset_all_weights()
    params = list(processor.parameters())
    print(DEVICE)
    print(processor)
    augmenting_path_network = None
    for key, algorithm in processor.algorithms.items():
        if type(algorithm) == models.BellFordNetwork:
            b_f_network = algorithm
    print(b_f_network)

    BATCH_SIZE = hyperparameters["batch_size"]
    PATIENCE_LIMIT = hyperparameters["patience_limit"]
    GROWTH_RATE = hyperparameters["growth_rate_sigmoid"]
    SIGMOID_OFFSET = hyperparameters["sigmoid_offset"]

    patience = 0
    # last_mean = 0
    # last_final = 0
    # last_loss = 0*1e9 if augmenting_path_network is not None else 1e9
    # cnt = 0
    best_final_acc = 0
    best_mean_acc = 0
    best_loss = np.inf

    fmt = get_print_format()

    best_model = models.AlgorithmProcessor(
        DIM_LATENT, BellmanFordDataset, args["--processor-type"]
    ).to(DEVICE)
    best_model.algorithms = nn.ModuleDict(processor.algorithms.items())
    best_model.load_state_dict(copy.deepcopy(processor.state_dict()))

    torch.set_printoptions(precision=20)

    with torch.autograd.profiler.profile(enabled=False, use_cuda=True) as prof:
        optimizer = optim.Adam(
            params,
            lr=hyperparameters["lr"],
            weight_decay=hyperparameters["weight_decay"],
        )
        for epoch in range(3000):  # FIXME
            if interrupted():
                break
            # 0.0032
            processor.train()
            iterate_over(processor, optimizer)

            patience += 1
            print("Epoch {:4d}: \n".format(epoch), end=" ")
            processor.eval()
            iterate_over(processor)
            (
                total_loss_dist,
                total_loss_pred,
                total_loss_term,
                total_loss,
                mean_step_acc,
                final_step_acc,
            ) = get_print_info(processor.algorithms["BellmanFord"])
            if get_hyperparameters()["calculate_termination_statistics"]:  # DEPRECATED
                print(
                    "Term precision:",
                    processor.algorithms["BellmanFord"].true_positive
                    / (
                        processor.algorithms["BellmanFord"].true_positive
                        + processor.algorithms["BellmanFord"].false_positive
                    )
                    if processor.algorithms["BellmanFord"].true_positive
                    + processor.algorithms["BellmanFord"].false_positive
                    else "N/A",
                )
                print(
                    "Term recall:",
                    processor.algorithms["BellmanFord"].true_positive
                    / (
                        processor.algorithms["BellmanFord"].true_positive
                        + processor.algorithms["BellmanFord"].false_negative
                    )
                    if processor.algorithms["BellmanFord"].true_positive
                    + processor.algorithms["BellmanFord"].false_negative
                    else "N/A",
                )

            if final_step_acc >= best_final_acc:
                best_final_acc = final_step_acc
                best_model.load_state_dict(copy.deepcopy(processor.state_dict()))
                patience = 0
            else:
                patience += 1

            print(
                fmt.format(
                    mean_step_acc,
                    final_step_acc,
                    total_loss_dist,
                    total_loss_pred,
                    total_loss_term,
                    total_loss,
                    patience,
                )
            )

            os.makedirs(f"checkpoints/{NAME}", exist_ok=True)
            torch.save(
                processor.state_dict(),
                f"checkpoints/{NAME}/test_{NAME}_epoch_" + str(epoch) + ".pt",
            )

            if patience >= PATIENCE_LIMIT:
                break

    torch.save(best_model.state_dict(), f"checkpoints/{NAME}/best_{NAME}.pt")
