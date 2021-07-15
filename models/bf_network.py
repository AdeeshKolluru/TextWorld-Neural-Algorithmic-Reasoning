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


class BellFordNetwork(AlgorithmBase):
    def __init__(self, latent_features, node_features, edge_features, algo_processor, dataset_class, dataset_root, bias=False, use_ints=False, bits_size=None):
        super(BellFordNetwork, self).__init__(latent_features, node_features, edge_features, bits_size if use_ints else 1, algo_processor, dataset_class, dataset_root, bias=bias)
        self.bits_size = bits_size
        if use_ints:
            self.bit_encoder = nn.Sequential(
                nn.Linear(bits_size, latent_features, bias=bias),
                nn.LeakyReLU()
            )

        ne_input_features = 2*latent_features if use_ints else node_features+latent_features
        self.node_encoder = nn.Sequential(
            nn.Linear(ne_input_features, latent_features, bias=bias),
            nn.LeakyReLU()
        )

        ee_input_features = 2*latent_features if use_ints else edge_features
        self.edge_encoder = nn.Sequential(
            nn.Linear(ee_input_features, latent_features, bias=bias),
            nn.LeakyReLU()
        )

        self.pred_network = PredecessorNetwork(latent_features, latent_features, bias=bias)
        if not use_ints:
            self.infinity = nn.Parameter(torch.randn(latent_features))

    def zero_tracking_losses_and_statistics(self):
        super().zero_tracking_losses_and_statistics()
        self.losses = {
            "pred": 0,
            "dist": 0,
            "term": 0,
        }
        self.predictions = {
            "terminations": [],
            "distances": [],
            "predecessors": []
        }
        self.actual = {
            "terminations": [],
            "distances": [],
            "predecessors": []
        }


    def get_input_output_features(self, batch, SOURCE_NODES):
        x = batch.x.clone()
        y = batch.y.clone()
        mask_x = BellFordNetwork.get_input_infinity_mask(batch.x)
        mask_y = BellFordNetwork.get_input_infinity_mask(batch.y)
        x[:, :, 0] += SOURCE_NODES[batch.batch].unsqueeze(1)
        y[:, :, 0] += SOURCE_NODES[batch.batch].unsqueeze(1)
        x[:, :, 1][mask_x] = -1
        y[:, :, 1][mask_y] = -1
        return x, y
    
    def get_input_infinity_mask(inp):
        mask = inp[:, :, 1] == 255
        return mask
    
    def loop_body(self, batch, inp, true_termination, to_process,
            compute_losses_and_broken, enforced_mask, GRAPH_SIZES):
        assert not self.training or to_process.any()
        assert self.mask_cp.any()
        iimask = BellFordNetwork.get_input_infinity_mask(batch.x)[:, 0]
        current_latent, distances, predecessors_p, continue_p = \
            self(
                batch.batch,
                GRAPH_SIZES,
                inp,
                self.last_latent,
                batch.edge_index,
                batch.edge_attr,
                iimask
            )
        self.update_states(distances, predecessors_p,
                           continue_p, current_latent)
        start = time.time()

        self.aggregate_loss_steps_and_acc(
            batch.batch, self.mask, self.mask_cp,
            compute_losses_and_broken,
            self.last_continue_p, true_termination,
            self.y_curr,
            distances, predecessors_p)
        
    def update_states(self, distances, predecessors_p,
                      continue_p, current_latent):
        super().update_states(continue_p, current_latent)
        DIM_LATENT = get_hyperparameters()["dim_latent"]
        # self.last_distances = torch.where(self.mask, utils.bit2integer(distances), self.last_distances)
        self.last_distances = torch.where(self.mask, distances, self.last_distances)
        self.last_predecessors_p = torch.where(self.mask, predecessors_p, self.last_predecessors_p)
        self.last_output = self.last_distances
        
    
    def set_initial_last_states(self, batch, STEPS_SIZE, SOURCE_NODES): 
        hyperparameters = get_hyperparameters()
        DEVICE = hyperparameters["device"]
        DIM_LATENT = hyperparameters["dim_latent"]
        DIM_NODES = hyperparameters["dim_nodes_BellmanFord"]
        DIM_EDGES = hyperparameters["dim_edges_BellmanFord"]

        SIZE = batch.num_nodes
        super().set_initial_last_states(batch, STEPS_SIZE, SOURCE_NODES)
        self.last_predecessors_p = torch.full((SIZE, SIZE), -1e9, device=DEVICE)
        self.last_predecessors_p[(SOURCE_NODES, SOURCE_NODES)] = 1e9
        self.last_distances = self.last_output.clone()
        self.last_distances[SOURCE_NODES] = 0.
        
    def aggregate_loss_steps_and_acc(
            self,
            batch_ids, mask, mask_cp,
            compute_losses_and_broken,
            continue_p, true_termination,
            y_curr,
            distances, predecessors_p):
        loss_dist, loss_pred, loss_term, steps, processed_nodes, step_acc =\
            self.get_step_loss(
                batch_ids, mask, mask_cp,
                y_curr,
                continue_p, true_termination,
                distances, predecessors_p,
                compute_losses_and_broken=compute_losses_and_broken)

        self.losses["dist"] += loss_dist
        self.losses["pred"] += loss_pred
        self.losses["term"] += loss_term
        self.aggregate_steps(steps, processed_nodes)
        
    
    def get_real_output_values(y_curr):
        DEVICE = 'cuda' if y_curr.is_cuda else 'cpu'
        zero_selector = torch.tensor([0], dtype=torch.long, device=DEVICE)
        one_selector = torch.tensor([1], dtype=torch.long, device=DEVICE)
        distances_real = torch.index_select(y_curr, 1, one_selector).squeeze()
        predecessors_real = torch.index_select(y_curr, 1, zero_selector).long().squeeze()
        return distances_real, predecessors_real
    
    
    def mask_infinities(mask, y_curr, distances, predecessors_p):
        distances_real, predecessors_real = BellFordNetwork.get_real_output_values(y_curr)
        non_inf_indices = (predecessors_real[mask] != 255).nonzero().squeeze()
        distances = torch.index_select(distances[mask], 0, non_inf_indices)
        predecessors_p_masked = torch.index_select(predecessors_p[mask], 0, non_inf_indices)
        distances_real = torch.index_select(distances_real[mask], 0, non_inf_indices)
        predecessors_real = torch.index_select(predecessors_real[mask], 0, non_inf_indices)
        return distances, distances_real, predecessors_p_masked, predecessors_real

    
    def get_step_loss(self,
                      batch_ids, mask, mask_cp,
                      y_curr,
                      continue_p, true_termination,
                      distances, predecessors_p,
                      compute_losses_and_broken=True):
        distances_masked, distances_real_masked, predecessors_p_masked, predecessors_real_masked = \
                BellFordNetwork.mask_infinities(mask, y_curr, distances, predecessors_p)
        steps = sum(mask_cp.float())

        
        train = self.training

        loss_dist, loss_pred, loss_term, processed_nodes, step_acc = 0, 0, 0, 0, 1
        if distances_real_masked.nelement() != 0 and compute_losses_and_broken:
            processed_nodes = len(distances_real_masked)
            if self.bits_size is None:
                loss_dist = F.mse_loss(distances_masked, distances_real_masked, reduction='sum')
            else:
                loss_dist = F.binary_cross_entropy_with_logits(distances_masked, utils.integer2bit(distances_real_masked), reduction='sum')
            loss_pred = F.cross_entropy(predecessors_p_masked, predecessors_real_masked, ignore_index=-1, reduction='sum')

        if compute_losses_and_broken:
            assert mask_cp.any(), mask_cp
            loss_term = F.binary_cross_entropy_with_logits(continue_p[mask_cp], true_termination[mask_cp], reduction='sum', pos_weight=torch.tensor(1.00))
            if get_hyperparameters()["calculate_termination_statistics"]:
                self.update_termination_statistics(continue_p[mask_cp], true_termination[mask_cp])

            assert loss_term.item() != float('inf')
            
        if not train and mask_cp.any() and compute_losses_and_broken:
            assert mask_cp.any()
            _, predecessors_real = BellFordNetwork.get_real_output_values(y_curr)
            predecessors_p_split = utils.split_per_graph(batch_ids, predecessors_p)
            predecessors_real_split = utils.split_per_graph(batch_ids, predecessors_real)
            correct, tot = BellFordNetwork.calculate_step_acc(torch.max(predecessors_p_split[mask_cp], dim=2).indices, predecessors_real_split[mask_cp])
            self.mean_step.extend(correct/tot.float())
            step_acc = correct/tot.float()
            
        return loss_dist, loss_pred, loss_term, steps, processed_nodes, step_acc
    
    

    def get_outputs(self, batch, adj_matrix, flow_matrix, compute_losses_and_broken):# Also updates broken invariants
        predecessors = torch.max(self.last_predecessors_p, dim=1).indices
        # if not self.training and compute_losses_and_broken:
        #     self.update_broken_invariants(batch, predecessors, adj_matrix, flow_matrix)
        return predecessors
    
    

    @overrides
    def get_losses_dict(self):
        #print("bits_size", self.bits_size)
        denom = self.sum_of_processed_nodes
        #if self.bits_size is not None:
            #denom *= self.bits_size
        return {
            "dist": self.losses["dist"] / float(denom) if self.sum_of_steps != 0 else 0,
            "pred": self.losses["pred"] / self.sum_of_processed_nodes if self.sum_of_steps != 0 else 0,
            "term": self.losses["term"] / self.sum_of_steps if self.sum_of_steps != 0 else 0,
        }


    @overrides
    def get_training_loss(self):
        return sum(self.get_losses_dict().values())

    @overrides
    def get_validation_losses(self):
        denom = self.validation_sum_of_processed_nodes
        if self.bits_size is not None:
            denom *= self.bits_size
        dist = self.validation_losses["dist"] / float(denom) if self.sum_of_steps != 0 else 0
        pred = self.validation_losses["pred"] / float(self.validation_sum_of_processed_nodes) if self.sum_of_steps != 0 else 0
        term = self.validation_losses["term"] / float(self.validation_sum_of_steps) if self.sum_of_steps != 0 else 0
        return dist, pred, term
    
    
    @overrides
    def get_validation_accuracies(self):
        return (sum(self.mean_step)/len(self.mean_step),
                sum(self.last_step)/self.last_step_total.float())

    @staticmethod
    def get_losses_from_predictions(predictions, actual):
        for key in predictions:
            if not isinstance(predictions[key], list):
                continue
            predictions[key] = torch.stack(predictions[key], dim=0)
            actual[key] = torch.stack(actual[key], dim=0)

        total_loss_reachability = F.binary_cross_entropy_with_logits(predictions["reachabilities"], actual["reachabilities"])
        total_loss_term = F.binary_cross_entropy_with_logits(predictions["terminations"], actual["terminations"])
        return total_loss_reachability, total_loss_term

    def zero_validation_stats(self):
        super().zero_validation_stats()
        self.validation_losses = {
            "pred": 0,
            "dist": 0,
            "term": 0,
        }

    def update_validation_stats(self, batch, predecessors):
        _, SOURCE_NODES = utils.get_sizes_and_source(batch)
        _, y = self.get_input_output_features(batch, SOURCE_NODES)
        predecessors_real = y[:, -1, 0]

        super().aggregate_last_step(predecessors, predecessors_real.squeeze())
        for key in self.validation_losses:
            self.validation_losses[key] += self.losses[key]
            
    def encode_edges(self, edge_attr):
        if self.bits_size is not None:
            edge_attr_w = self.bit_encoder(utils.integer2bit(edge_attr[:, 0]))
            edge_attr_cap = self.bit_encoder(utils.integer2bit(edge_attr[:, 1]))
            edge_attr = torch.cat((edge_attr_w, edge_attr_cap), dim=1)
        encoded_edges = self.edge_encoder(edge_attr)
        return encoded_edges

    def forward(self, batch_ids, GRAPH_SIZES, current_input, last_latent, edge_index, edge_attr, iimask, edge_mask=None):
        SIZE = last_latent.shape[0]
        if self.bits_size is not None:
            current_input = utils.integer2bit(current_input, self.bits_size)
            current_input = self.bit_encoder(current_input)
        else:
            current_input = current_input.unsqueeze(1)

        inp = torch.cat((current_input, last_latent), dim=1)

        encoded_nodes = self.node_encoder(inp)
        if self.steps == 0 and self.bits_size is None: # if we are not using integers we learn infinity embedding for the first step
            encoded_nodes[iimask] = self.infinity
        #print(edge_attr.shape)
        #print(edge_attr)
        edge_attr = edge_attr.unsqueeze(dim=-1)
        #print(edge_attr.shape)
        encoded_edges = self.encode_edges(edge_attr)

        latent_nodes = self.processor(encoded_nodes, encoded_edges, utils.flip_edge_index(edge_index))
        predecessors = self.pred_network(encoded_nodes, latent_nodes, encoded_edges, edge_index)

        output = self.decoder_network(torch.cat((encoded_nodes, latent_nodes), dim=1))
        distances = output.squeeze()
        continue_p = self.get_continue_p(batch_ids, latent_nodes, GRAPH_SIZES)
        return latent_nodes, distances, predecessors, continue_p

