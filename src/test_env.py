import gym
import numpy as np

from envs import BoardEnv


# list of source positions
sources = np.array([[0,0]])

# list of target positions
targets = np.array([[5,0]])

# list of obstacles
obstacles = []

# BoardEnv initialization
env = BoardEnv(sources, targets, obstacles)
env.reset()

done = False

while not done:
    action = 0  # <--- this is where the magic should happen
    state, reward, done, _ = env.step(action)
    print(state)

