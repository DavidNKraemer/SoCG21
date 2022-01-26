import numpy as np
import argparse
import yaml
import os
import stable_baselines3 as sb
from copy import deepcopy
from stable_baselines3.common.callbacks import EvalCallback

from src.envs import board


if __name__ == "__main__":

    save_dir = 'test_logs'
    config = 'dqn_config.yml'

    with open(config, "r") as f:
        config = yaml.safe_load(f)


    # set up environment
    reshaper = lambda x: np.reshape(x, (-1, 2))
    groups = ['starts', 'targets', 'obstacles']
    for group in groups:
        data = config['env_config'][group]
        config['env_config'][group] = reshaper(np.array(data))

    max_timesteps = config["env_config"]["max_timesteps"]

    env = board.env(
        *(config['env_config'][g] for g in groups),
        max_timesteps=max_timesteps
    ).unwrapped.to_gym()

    model = sb.DQN.load(f"{save_dir}/best_model.zip")

    # Enjoy trained agent
    state = env.reset()
    done = False
    while not done:
        action, _states = model.predict(state)
        state, reward, done, info = env.step(action)
        env.render()

    input('Press anything to close')
    env.close()

