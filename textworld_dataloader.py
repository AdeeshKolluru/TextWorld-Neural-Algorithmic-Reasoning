from coincollector_mazes.get_maze import get_maze
import torch_geometric
from torch_geometric.utils.convert import from_networkx
import random
from textworld_helpers.generic import to_np, to_pt, preproc, _words_to_ids, pad_sequences, max_len
import numpy as np

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

# creating a word vocab - unique list of words from the sentences above
word_vocab = []
for data in data_lst:
    for i in range(len(data.description)):
        word_vocab.extend(data.description[i].split())
word_vocab = np.unique(word_vocab)

# convert word to id
word2id = {w: i for i, w in enumerate(word_vocab)}

# convert observation to int and pad
for data in data_lst:
    observation_id_list = []
    for i in range(len(data.description)):
        observation_list = data.description[i].split()
        observation_id_list.append([word2id[w] for w in observation_list])
    observation_id_list = pad_sequences(observation_id_list, maxlen=max_len(observation_id_list), padding='post').astype('int32')
    input_observation = to_pt(observation_id_list, False)
    data.input_observation = input_observation




