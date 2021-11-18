import os
import sys

sys.path.append(
    '..'
)

import numpy as np
from src.envs import PZBoardEnv, bot_reward
from IPython import embed


li = 0


def left_policy():
    global li
    out = 2 * (li % 2)
    li += 1
    return out


ri = 0


def right_policy():
    global ri
    if ri % 2 == 0:
        ri += 1
        return 1
    else:
        ri += 1
        return 2

rule = {0: left_policy, 1: right_policy}


if __name__ == "__main__":
    starts = np.array([[0, 0], [3, 0]])
    targets = np.array([[3, 3], [0, 3]])
    obstacles = np.array([])
    env = PZBoardEnv(starts, targets, obstacles, 0, lambda bot: 1)
    env.reset()

    for i, agent in enumerate(env.agent_iter()):
        print(i, agent)
        observation, reward, done, info = env.last()

        action = rule[int(agent[-1])]()

        env.step(action)
