import gym
import numpy as np

from envs import BoardEnv

sources = np.array([
    [0,0]
])

targets = np.array([
    [5,0]
])

obstacles = []

env = BoardEnv(sources, targets, obstacles)
env.reset()

terminated = False

while not terminated:
    action = 0
    state, reward, terminated, _ = env.step(action)
    print(state)

