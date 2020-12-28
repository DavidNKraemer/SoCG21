import argparse

import random
import time

from itertools import product, chain
from random import choice, randint


def sign(x):
    return {True: 1, False: -1}[x > 0] if x else x


class Agent:

    def __init__(self, start, target):
        self.start = start
        self.target = target

    def reset(self):
        self.pos = self.start

    def priority(self):
        return 0

    def move(self):
        if self.isdone():
            return 0

        vert = sign(self.target[0] - self.pos[0])
        horiz = sign(self.target[1] - self.pos[1])

        moves = [(i,j) for i, j in product([0,vert], [0,horiz]) if abs(i)+abs(j) == 1]
        
        chosen = choice(moves)
        self.pos = (self.pos[0] + chosen[0], self.pos[1] + chosen[1])

        return abs(chosen[0]) + abs(chosen[1])

    def isdone(self):
        return self.pos == self.target


class GameBoard:

    def __init__(self, starts, targets):
        self.starts = starts
        self.targets = targets

        self.row_range = [
            min(map(lambda pair: pair[0], chain(self.starts, self.targets))),
            max(map(lambda pair: pair[0], chain(self.starts, self.targets)))
        ]
        self.col_range = [
            min(map(lambda pair: pair[1], chain(self.starts, self.targets))),
            max(map(lambda pair: pair[1], chain(self.starts, self.targets)))
        ]
        
    def reset(self):
        self.agents = list(map(lambda tup: Agent(*tup), zip(self.starts, self.targets)))
        self.time = 0
        self.dist = 0

        for agent in self.agents:
            agent.reset()

    def queue(self):
        return sorted(
            self.agents, key=lambda agent: agent.priority(), reverse=True
        )
        
    def step(self):
        queue = self.queue()
        for i, agent in enumerate(queue):
            self.dist += agent.move()
        self.time += 1

    def isdone(self):
        return all(agent.isdone() for agent in self.agents)

    def __str__(self):
        width = len(str(len(self.agents)))
        board = [[" " * (width+1) for _ in range(self.col_range[0], self.col_range[1]+1)] \
                 for _ in range(self.row_range[0], self.row_range[1]+1)]

        all_locations = zip(
            self.starts, self.targets, map(lambda agent: agent.pos, self.agents)
        )
        for i, locations in enumerate(all_locations):
            for c, loc in zip(['S','T','A'], locations):
                row, col = loc
                board[row-self.row_range[0]][col-self.col_range[0]] = "{}{:0{}d}".format(c, i, width)

        str_board = ['\u2502' + "".join(x for x in row) + '\u2502' for row in board]
        upper_side = '\u250c' + '\u2500' * (len(str_board[0]) - 2) + '\u2510\n'
        lower_side = '\n\u2514' + '\u2500' * (len(str_board[0]) - 2) + '\u2518\n'
        
        return upper_side + '\n'.join(str_board) + lower_side


if __name__ == '__main__':
    nrows, ncols, nagents = 15, 40, 10

    starts =  [(randint(0,nrows), randint(0,ncols)) for _ in range(nagents)]
    targets = [(randint(0,nrows), randint(0,ncols)) for _ in range(nagents)]
    
    board = GameBoard(starts, targets)
    board.reset()

    while not board.isdone():
        print(chr(27) + "[2J")
        print(board)
        print(f"Time elapsed: {board.time}")
        print(f"Distance traveled: {board.dist}")
        board.step()
        time.sleep(0.1)
    print(board)
    print(f"Time elapsed: {board.time}")
    print(f"Distance traveled: {board.dist}")

