import numpy as np
from agent import *

# instantiate Agents
a = Agent(np.array([0, 0]), np.array([5, 5]))
b = Agent(np.array([5, 0]), np.array([0, 5]))
c = Agent(np.array([0, 5]), np.array([5, 0]))
d = Agent(np.array([5, 5]), np.array([0, 0]))

# pre-process local neighborhoods using magic in DistributedState's __init__
board = DistributedBoard(set([a, b, c, d]), obstacles)
policy = Policy(board)

# move an agent... God told us the best move to make (how nice of Him!)
board.direct(b, "N")  # affected Agents' local neighborhoods updated magically

##################################
while not board.isdone():
    # priority queue to find which move to make (God hides his face)
    board.populate_queue()
    for agent in board.queue:
        # get instruction
        direction = policy(agent)
        board.direct(agent, direction)

# key objects: Agent, DistributedBoard, Policy, (priority_queue?)
