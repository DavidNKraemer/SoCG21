# check our env for compatibility with stable_baselines3

import numpy as np
from src.envs import board

from stable_baselines3.common.env_checker import check_env


if __name__ == "__main__":

    starts = np.array([[0, 0], [3, 0]])
    targets = np.array([[3, 0], [0, 0]])
    obstacles = np.array([[1, 1]])
    env = board.env(starts, targets, obstacles).unwrapped.to_gym()
    env.reset()

    check_env(env)
