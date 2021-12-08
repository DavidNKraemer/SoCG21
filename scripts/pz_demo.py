import sys

sys.path.append("/home/logan/Documents/Code/SoCG21/")

import numpy as np
from src.envs import board


if __name__ == "__main__":
    starts = np.array([[0, 0], [3, 0]])
    targets = np.array([[3, 0], [0, 0]])
    obstacles = np.array([[1, 1]])
    env = board.env(starts, targets, obstacles)
    env.reset()

    # iterate over agents in the order in which they move
    for i, agent in enumerate(env.agent_iter(max_iter=20)):
        print(f"Agent: {agent}; Iteration: {i}")
        observation, reward, done, info = env.last()
        # print(observation)
        print(env._cumulative_rewards)

        if i % 2 == 0:
            # East
            action = 0
        else:
            # West
            action = 1

        env.step(action)
