import sys

sys.path.append("/home/logan/Documents/Code/SoCG21/")

import numpy as np
from src.envs import board


if __name__ == "__main__":
    starts = np.array([[0, 0], [4, 0]])
    targets = np.array([[4, 0], [0, 0]])
    obstacles = np.array([[1, 1]])
    env = board.env(starts, targets, obstacles)
    env.reset()

    for i, agent in enumerate(env.agent_iter(max_iter=20)):
        print(f"Agent: {agent}; Iteration: {i}")
        observation, reward, done, info = env.last()
        print(observation)
        print(reward)

        if i % 2 == 0:
            action = 0
        elif i % 2 == 1:
            action = 1
        else:
            raise Exception("What the heck?")

        env.step(action)
