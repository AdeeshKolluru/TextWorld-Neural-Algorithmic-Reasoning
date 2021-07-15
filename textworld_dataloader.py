from coincollector_mazes.get_maze import get_maze
import torch_geometric
from torch_geometric.utils.convert import from_networkx
import random
from textworld_helpers.generic import to_np, to_pt, preproc, _words_to_ids, pad_sequences, max_len

# set random seeds and nodes for graph generation
random.seed(0)
random_nodes = random.sample(range(5,30), 10)
random_seeds = random.sample(range(0,100), 10)

# graph generation and conversion to torch geometric
data_lst = []
for nodes, seeds in zip(random_nodes, random_seeds):
    G = get_maze(nodes, seeds)
    data = from_networkx(G)
    data_lst.append(data)
    
# creating a word vocab
# TODO 

# convert word to id
word2id = {w: i for i, w enumerate(word_vocab)}

# convert observation to int and pad
# TODO create observation id list - list of integers
observation_id_list = pad_sequences(observation_id_list, maxlen=max_len(observation_id_list), padding='post').astype('int32'))
input_observation = to_pt(observation_id_list, False)



