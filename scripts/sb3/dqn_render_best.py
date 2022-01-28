import argparse
import numpy as np
import os
import stable_baselines3 as sb
import time
import yaml

from copy import deepcopy
from src.envs import board
from stable_baselines3.common.callbacks import EvalCallback

parser = argparse.ArgumentParser("python dqn_render_best.py")

parser.add_argument(
    "--logdir", type=str, default=None, required=True,
    help="Directory to read trial data from"
)
parser.add_argument(
    "--config", type=str, default=None, required=True,
    help="Filename of configuration file"
)

args = parser.parse_args()


if __name__ == "__main__":


    with open(args.config, "r") as f:
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

    model_file = os.path.join(args.logdir, "best_model.zip")
    model = sb.DQN.load(model_file)

    # Enjoy trained agent
    state = env.reset()
    done = False
    while not done:
        action, _states = model.predict(state)
        state, reward, done, info = env.step(action)
        env.render()
        time.sleep(0.5)

    input('Press [ENTER] to close')
    env.close()

