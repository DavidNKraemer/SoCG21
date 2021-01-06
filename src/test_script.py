import numpy as np
from board import *

np.random.seed(0)

# instantiate Agents
sources = np.random.randint(-10,10, size=(4,2))
targets = np.random.randint(-10,10, size=(4,2))

# pre-process local neighborhoods using magic in DistributedState's __init__
board = DistributedBoard(sources, targets, [])
policy = DumbPolicy(board)

##################################
t = 0
while not board.isdone():
    # priority queue to find which move to make (God hides his face)
    print(f't = {t}')
    agent = board.pop()
    direction = policy(agent)
    agent.move(direction)
    board.insert(agent)
    t += 1

# key objects: Agent, DistributedBoard, Policy, (priority_queue?)
