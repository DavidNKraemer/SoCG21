import numpy as np
import argparse
import yaml
import os
import stable_baselines3 as sb
from copy import deepcopy
from stable_baselines3.common.callbacks import EvalCallback

from src.envs import board


parser = argparse.ArgumentParser("python dqn_train.py")

parser.add_argument(
    "--logdir", type=str, default=None, required=True,
    help="Directory to store trial data in"
)
parser.add_argument(
    "--config", type=str, default=None, required=True,
    help="Filename of configuration file"
)

args = parser.parse_args()


if __name__ == "__main__":

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    logdir = args.logdir if args.logdir is not None else config["logdir"]
    os.makedirs(logdir, exist_ok=True)

    # set up environment
    reshaper = lambda x: np.reshape(x, (-1, 2))
    starts = reshaper(np.array(config["env_config"]["starts"]))
    targets = reshaper(np.array(config["env_config"]["targets"]))
    obstacles = reshaper(np.array(config["env_config"]["obstacles"]))
    max_timesteps = config["env_config"]["max_timesteps"]
    env = board.env(
        starts, targets, obstacles, max_timesteps=max_timesteps
    ).unwrapped.to_gym()

    # set up model
    n_eval_episodes = config["training_config"]["n_eval_episodes"]
    eval_freq = config["training_config"]["eval_freq"]
    best_model_save_path = logdir
    log_path = logdir
    deterministic_eval = config["training_config"]["deterministic_eval"]
    render = config["training_config"]["render"]
    eval_env = deepcopy(env)

    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=n_eval_episodes,
        eval_freq=eval_freq,
        best_model_save_path=best_model_save_path,
        log_path=log_path,
        deterministic=deterministic_eval,
        render=render,
    )
    model = sb.DQN("MultiInputPolicy", env, **config["model_kwargs"])

    # training
    n_training_steps = config["training_config"]["n_training_steps"]
    model.learn(n_training_steps, callback=eval_callback)
