import random

from coincollector_mazes.get_maze import get_maze, print_maze_info


def generate_maze_data_with_seed(seed: int, number_of_mazes: int) -> list:
    random.seed(seed)
    maze_list = []
    for k in range(number_of_mazes):
        print('Generating maze %s of %s' % (k, number_of_mazes))
        this_seed = random.randint(1,100000)
        maze_list += [get_maze(250, this_seed)]
    return maze_list


if __name__ == '__main__':
    number_of_mazes = 10
    maze_list = generate_maze_data_with_seed(100, number_of_mazes)
    print_maze_info(maze_list[0])
    print(len(maze_list))
