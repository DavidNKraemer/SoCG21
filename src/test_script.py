import numpy as np
from board import *

np.random.seed(0)

# instantiate Agents
sources = np.random.randint(-10, 10, size=(20,2))
targets = np.random.randint(-10, 10, size=(20,2))
obstacles = np.random.randint(-10, 10, size=(15,2))

# pre-process local neighborhoods using magic in DistributedState's __init__
board = DistributedBoard(sources, targets, obstacles)
policy = DumbPolicy(board)

##################################
t = 0
while not board.isdone():
    # priority queue to find which move to make (God hides his face)
    print(f't = {t}')
    agent = board.pop()
    print(f'local state for moving agent = {agent.state}')
    direction = policy(agent)
    agent.move(direction)
    board.insert(agent)
    t += 1

# key objects: Agent, DistributedBoard, Policy, (priority_queue?)
