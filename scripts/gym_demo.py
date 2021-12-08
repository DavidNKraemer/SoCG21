import sys

sys.path.append("/home/logan/Documents/Code/SoCG21/")

import numpy as np
from src.envs import board


if __name__ == "__main__":
    starts = np.array([[0, 0], [3, 0]])
    targets = np.array([[3, 0], [0, 0]])
    obstacles = np.array([[1, 1]])
    env = board.env(starts, targets, obstacles).unwrapped.to_gym()
    state = env.reset()

    n_steps = 100
    for step in range(n_steps):
        # next_state, reward, done, info = env.step({"bot_0": 0, "bot_1": 1})
        next_state, reward, done, info = env.step(env.action_space.sample())

        print(f"iteration: {step}; reward: {reward}")
        if done:
            break

        state = next_state
