import networkx as nx
from textworld.challenges.tw_coin_collector.coin_collector import make as make_coin_collector_game
import textworld
from textworld.generator.game import GameOptions


def build_game_options(seed: int) -> GameOptions:
    options = GameOptions()
    options.seeds = seed
    return options

def add_shortest_path_info(G: nx.DiGraph, game: textworld.Game) -> nx.DiGraph:
    # label all edges as not contained in shortest path
    for edge in G.edges:
        G.add_edge(edge[0], edge[1], contained_in_shortest_path=0)

    for node in G.nodes.items():
        if node[1]['is_starting_position'] == 1:
            starting_node = node[0]
        if node[1]['has_coin'] == 1:
            destination_node = node[0]
    print(starting_node)
    print(destination_node)
    shortest_path = nx.shortest_path(G, source=starting_node, target=destination_node)
    print(shortest_path)
    for k in range(len(shortest_path)-1):
        G.add_edge(shortest_path[k], shortest_path[k+1], contained_in_shortest_path=1)
    return G

def get_maze(level: int, seed: int) -> nx.DiGraph:
    '''
    Generates a graph model of the world of a Textworld Coin-Collector game.

    Args:
        level: Difficulty level of the game. See documentation of make_coin_collector_game for details. Gives a line graph for level <= 100 and graphs that are not line graphs for level >= 101.
        seed: Random seed used for generation of the graph model.

    Returns: A networkx DiGraph where nodes have node data
    'description', which is a string of the in-game room description;
    'is_starting_position', which 1 if the room is the unique room to be the player's starting position and 0 otherwise;
    'has_coin', which 1 if the room is the unique room to contain a coin and 0 otherwise;
    and edges have edge data
    'direction', which is 0 [1/2/3] if beginning of the edge is north [east/south/west] of end of the edge;
    'contained_in_shortest_path', if the edge is contained in a shortest path from the player starting location to the coin room. If more than one shortest path exists, one is chosen arbitrarily.
    '''
    options = build_game_options(seed)
    game = make_coin_collector_game({'level': level}, options)

    G = nx.DiGraph()

    for fact in game.world.facts:
        if fact.name == 'at' and fact.names[0] == 'P':
            player_start_room = fact.names[1]
        if fact.name == 'at' and fact.names[0] == 'o_0':
            coin_room = fact.names[1]

    for room in game.world._rooms:
        if room.name == player_start_room:
            is_starting_position = 1
        else:
            is_starting_position = 0

        if room.name == coin_room:
            has_coin = 1
        else:
            has_coin = 0

        G.add_node(
            room.name,
            description=game.infos[room.name].desc,
            is_starting_position=is_starting_position,
            has_coin=has_coin
        )

    for fact in game.world.facts:
        if fact.name in {'north_of', 'east_of', 'south_of', 'west_of'}:
            if fact.name == 'north_of':
                direc = 0
            elif fact.name == 'east_of':
                direc = 1
            elif fact.name == 'south_of':
                direc = 2
            elif fact.name == 'west_of':
                direc = 3
            G.add_edge(fact.names[0], fact.names[1], direction=direc)

    G = add_shortest_path_info(G, game)

    return G

def print_maze_info(G: nx.DiGraph):
    print('List of all nodes:')
    for node in G.nodes.items():
        print(node)

    print('List of all edges:')
    for edge in G.edges.data():
        print(edge)


if __name__ == '__main__':
    G = get_maze(5, 100)
    print_maze_info(G)
