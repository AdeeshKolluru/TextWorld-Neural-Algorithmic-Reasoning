import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric

from hyperparameters import get_hyperparameters
import utils
from layers import PredecessorNetwork, GAT
from models import AlgorithmBase
from overrides import overrides


class MazeNetwork(AlgorithmBase):
    def __init__(self, latent_features, node_features, edge_features, algo_processor, dataset_class, dataset_root, output_features=1, bias=False):
        super(MazeNetwork, self).__init__(latent_features, node_features, edge_features, output_features, algo_processor, dataset_class, dataset_root, bias=bias)

        ne_input_features = node_features+latent_features
        self.node_encoder = nn.Sequential(
            nn.Linear(ne_input_features, latent_features, bias=bias),
            nn.LeakyReLU()
        )

        ee_input_features = edge_features
        self.edge_encoder = nn.Sequential(
            nn.Linear(ee_input_features, latent_features, bias=bias),
            nn.LeakyReLU()
        )

        self.pred_network = PredecessorNetwork(latent_features, latent_features, bias=bias)
        
        self.infinity = nn.Parameter(torch.randn(latent_features))

    def zero_tracking_losses_and_statistics(self):
        super().zero_tracking_losses_and_statistics()
        self.losses = {
            "path": []
        }
        self.predictions = {
            "edge_mask": []
        }
        self.actual = {
            "contained_in_shortest_path": []
        }


    def get_input_output_features(self, batch, SOURCE_NODES):
        x = batch.x.clone()
        y = batch.y.clone()
        mask_x = MazeNetwork.get_input_infinity_mask(batch.x)
        mask_y = MazeNetwork.get_input_infinity_mask(batch.y)
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
        iimask = MazeNetwork.get_input_infinity_mask(batch.x)[:, 0]
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
        DIM_NODES = hyperparameters["dim_nodes_Maze"]
        DIM_EDGES = hyperparameters["dim_edges_Maze"]

        SIZE = batch.num_nodes
        super().set_initial_last_states(batch, STEPS_SIZE, SOURCE_NODES)
        self.last_predecessors_p = torch.full((SIZE, SIZE), -1e9, device=DEVICE)
        self.last_predecessors_p[(SOURCE_NODES, SOURCE_NODES)] = 1e9
        self.last_distances = self.last_output.clone()
        self.last_distances[SOURCE_NODES] = 0.

    
    def get_real_output_values(y_curr):
        DEVICE = 'cuda' if y_curr.is_cuda else 'cpu'
        zero_selector = torch.tensor([0], dtype=torch.long, device=DEVICE)
        one_selector = torch.tensor([1], dtype=torch.long, device=DEVICE)
        distances_real = torch.index_select(y_curr, 1, one_selector).squeeze()
        predecessors_real = torch.index_select(y_curr, 1, zero_selector).long().squeeze()
        return distances_real, predecessors_real
    

    def get_outputs(self, batch, adj_matrix, flow_matrix, compute_losses_and_broken):# Also updates broken invariants
        predecessors = torch.max(self.last_predecessors_p, dim=1).indices
        # if not self.training and compute_losses_and_broken:
        #     self.update_broken_invariants(batch, predecessors, adj_matrix, flow_matrix)
        return predecessors
    
    @staticmethod
    def get_adj_from_predecessors():
        # Seems not differentiable
        pass

    def get_maze_loss(self, batch):
        # Not verified, may need to change
        shortest_path_adj = utils.get_adj_matrix(batch.edge_index[batch.contained_in_shortest_path])


        return None

    @overrides
    def get_losses_dict(self):
        #print("bits_size", self.bits_size)
        denom = self.sum_of_processed_nodes
        #if self.bits_size is not None:
            #denom *= self.bits_size
        return {
            "path": self.losses["path"] / float(denom) if self.sum_of_steps != 0 else 0,
        }


    # @overrides
    # def get_training_loss(self):
    #     return sum(self.get_losses_dict().values())

    # @staticmethod
    # def get_losses_from_predictions(predictions, actual):
    #     for key in predictions:
    #         if not isinstance(predictions[key], list):
    #             continue
    #         predictions[key] = torch.stack(predictions[key], dim=0)
    #         actual[key] = torch.stack(actual[key], dim=0)

    #     total_loss_reachability = F.binary_cross_entropy_with_logits(predictions["reachabilities"], actual["reachabilities"])
    #     total_loss_term = F.binary_cross_entropy_with_logits(predictions["terminations"], actual["terminations"])
    #     return total_loss_reachability, total_loss_term

    def zero_validation_stats(self):
        super().zero_validation_stats()
        self.validation_losses = {
            "path": 0
        }

    # def update_validation_stats(self, batch, predecessors):
    #     _, SOURCE_NODES = utils.get_sizes_and_source(batch)
    #     _, y = self.get_input_output_features(batch, SOURCE_NODES)
    #     predecessors_real = y[:, -1, 0]

    #     super().aggregate_last_step(predecessors, predecessors_real.squeeze())
    #     for key in self.validation_losses:
    #         self.validation_losses[key] += self.losses[key]
            
    def encode_edges(self, edge_attr):
        encoded_edges = self.edge_encoder(edge_attr)
        return encoded_edges

    def forward(self, batch_ids, GRAPH_SIZES, current_input, last_latent, edge_index, edge_attr, iimask, edge_mask=None):
        SIZE = last_latent.shape[0]
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

        output = self.decoder_network(torch.cat((encoded_nodes, latent_nodes), dim=1))
        distances = output.squeeze()
        continue_p = self.get_continue_p(batch_ids, latent_nodes, GRAPH_SIZES)
        return latent_nodes, distances, predecessors, continue_p

