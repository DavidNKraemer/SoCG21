import gym
import numpy as np
import src.envs.board as board
import time

from src.envs.wrappers import gym_board_env

sources = np.array([
    [0, 0],
    [0, 10],
    [20,20]
])

targets = np.array([
    [10, 0],
    [0, 10],
    [40, 40]
])

obstacles = np.array([
    [1, 1],
    [9, 9],
    [5, 5]
])

env = gym_board_env(board.env(
    sources,
    targets,
    obstacles
))

n_runs = 100
n_steps = 100
for i in range(n_runs):
    env.reset()
    for step in range(n_steps):
        env.render()
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)

        if done:
            break

env.close()
