import networkx as nx
import random

from coincollector_mazes.get_maze import get_maze


def identify_dangling_rooms(G: nx.DiGraph) -> dict:
    '''
    Identifies rooms which have exactly one adjacent room.

    Args:
        G: A network-x DiGraph

    Returns:
        A dictionary with keys 0, 1, 2, 3, where key 0 [1/2/3] gives a list of string room names ('r_0', 'r_6' etc.) that have exactly one adjacent room and are north [east/south/west] of that adjacent room.
    '''
    dangling_rooms = {
        0: [],
        1: [],
        2: [],
        3: []
    }
    for node in G.nodes():
        out_edges = list(G.edges(node))

        if len(out_edges) == 1: # this means room is dangling
            dangling_direction = G.get_edge_data(*out_edges[0])['direction']
            dangling_rooms[dangling_direction] += [node]

    return dangling_rooms

def add_new_edges(G: nx.DiGraph, number_of_new_edges) -> nx.DiGraph:
    dangling_rooms = identify_dangling_rooms(G)
    while number_of_new_edges > 0:
        direction_choices = [0, 1, 2, 3]
        random.shuffle(direction_choices)
        for direction in direction_choices:
            random.shuffle(dangling_rooms[direction])
            if len(dangling_rooms[direction]) >= 2:
                G.add_edge(
                    dangling_rooms[direction].pop(),
                    dangling_rooms[direction].pop(),
                    
                )


if __name__ == '__main__':
    G = get_maze(210, 100)
    dangling_rooms = identify_dangling_rooms(G)
    print(dangling_rooms)

