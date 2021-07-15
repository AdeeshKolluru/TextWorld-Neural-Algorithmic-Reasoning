"""
Usage:
    test.py [options] [--algorithms=ALGO]... 

Options:
    -h --help                        Show this screen.
    --algorithms ALGO                Which algorithms to add. One of {AugmentingPath, BFS, BellmanFord}. [default: BellmanFord]
    --processor-type PROC            Type of processor. One of {MPNN, PNA, GAT}. [default: MPNN]    
    --scale UP                     Test on larger graph size. Remember to add underscore (e.g. _30) [default: ]
"""

import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch_geometric
from torch_geometric.data import DataLoader

from tqdm import tqdm
from docopt import docopt

from flow_datasets import SingleIterationDataset
from hyperparameters import get_hyperparameters
from train_bellmanford import iterate_over, load_algorithms, get_print_info
from flow_datasets import BellmanFordDataset
from models import AlgorithmProcessor, AugmentingPathNetwork


# args["--use-ints"] = True  # Always uses integers


def get_print_format():
    fmt = """
==========================
Mean step acc: {:.4f}    Last step acc: {:.4f}
loss-(dist,pred,term,total): {:.4f} {:.4f} {:.4f} {:.4f}
===============
"""
    return fmt

if __name__ == "__main__":
    args = docopt(__doc__)
    hyperparameters = get_hyperparameters()
    DEVICE = hyperparameters["device"]
    DIM_LATENT = hyperparameters["dim_latent"]

    processor = AlgorithmProcessor(
        DIM_LATENT, BellmanFordDataset, args["--processor-type"]
    ).to(DEVICE)
    load_algorithms(args["--algorithms"], processor)
    NAME = (
            'BellmanFord'+args["--processor-type"]+str(hyperparameters["test_lr"])+str(hyperparameters["test_weight_decay"])
        )
    processor.load_state_dict(torch.load(f'best_models/best_{NAME}.pt'))
    processor.eval()

    # upscale = args["--upscale"]

    for algorithm in processor.algorithms.values():
        algorithm.test_dataset = algorithm.dataset_class(
            algorithm.dataset_root, split="test"+args["--scale"], less_wired=True, device="cpu"
        )
        # print(algorithm.test_dataset[0].x[:, 0, 0])

    iterate_over(processor, test=True)
    # if "AugmentingPath" not in processor.algorithms:
    #     print("Mean/Last step acc", processor.algorithms["BFS"].get_validation_accuracies())
    #     exit(0)

    fmt = get_print_format()

    (
        total_loss_dist,
        total_loss_pred,
        total_loss_term,
        total_loss,
        mean_step_acc,
        final_step_acc,
    ) = get_print_info(processor.algorithms["BellmanFord"])


    if get_hyperparameters()["calculate_termination_statistics"]:
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

    print(
        fmt.format(
            mean_step_acc,
            final_step_acc,
            total_loss_dist,
            total_loss_pred,
            total_loss_term,
            total_loss,
        )
    )

# print(
#     fmt.format(
#         mean_step_acc,
#         final_step_acc,
#         tnr,
#         subtract_acc,
#         total_loss_dist,
#         total_loss_pred,
#         total_loss_term,
#         # find_min,
#         total_loss,
#         # sum(broken_invariants),
#         # len_broken,
#         # sum(broken_all),
#         # len_broken,
#         # sum(broken_reachabilities),
#         # len_broken,
#         # sum(broken_flows),
#         # len_broken,
#         "N/A",
#     )
# )
