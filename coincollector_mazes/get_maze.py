import networkx as nx
from textworld.challenges.tw_coin_collector.coin_collector import make as make_coin_collector_game
from textworld.generator.game import GameOptions


def build_game_options(seed: int) -> GameOptions:
    options = GameOptions()
    options.seeds = seed
    return options


def get_maze(level: int, seed: int) -> nx.DiGraph:
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
