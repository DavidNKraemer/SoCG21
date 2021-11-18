import sys

sys.path.append("/home/logan/Documents/Code/SoCG21/")

import numpy as np
from src.envs import PZBoardEnv


def left_policy():
    if np.random.uniform() > 0.5:
        return 0
    else:
        return 4


def right_policy():
    if np.random.uniform() > 0.5:
        return 1
    else:
        return 4


rule = {0: left_policy, 1: right_policy}


if __name__ == "__main__":
    starts = np.array([[0, 0], [3, 0]])
    targets = np.array([[3, 0], [0, 0]])
    obstacles = np.array([])
    env = PZBoardEnv(starts, targets, obstacles, 0, lambda bot: 1)
    env.reset()

    for i, agent in enumerate(env.agent_iter(max_iter=20)):
        print(f"Agent: {agent}; Iteration: {i}")
        print(env.observations)
        observation, reward, done, info = env.last()

        action = rule[int(agent[-1])]()

        env.step(action)
