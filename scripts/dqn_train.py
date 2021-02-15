import numpy as np
import datetime
import pickle
import os
import argparse
from shutil import copyfile

import src.dqn.trainer as trainer
from scripts.training_sequence import training_plan, basic_training

training_seq = basic_training['obstacles']
myopia_rate = 1
len_epoch = 1


# This is the function that gets all the env_tuples to train on
# NOTE: it is a very important function
def get_env_tuples():
    """
    Return a list of env_tuples to be trained on.
    """

    env_tuples = list(elem[1:] for elem in training_plan(
        training_seq, myopia_rate, len_epoch))

    return env_tuples


parser = argparse.ArgumentParser(
    'python dqn_train.py'
)
parser.add_argument('--output_dir', type=str,
                    default=None,
                    help='Directory to store trial data in')
parser.add_argument('--config', type=str,
                    default=None,
                    help='Name of YAML file containing agent configs')
parser.add_argument('--num_episodes', type=int,
                    default=None,
                    help='Number of training episodes')
parser.add_argument('--episode_length', type=int,
                    default=None,
                    help='Length of each episode')
parser.add_argument('--load_checkpoint', type=str,
                    default=None,
                    help='Specify checkpoint to be loaded')
parser.add_argument('--continue_training',
                    default=True,
                    help='If loading checkpoint, decide whether to continue training')
args = parser.parse_args()


plot_filename = 'plot.pdf'
checkpoint_filename = 'checkpoint.pt'


def pickle_obj(filename, obj):
    """
    Pickle an object into the specified filename.
    """

    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def unpickle_obj(filename):
    """
    Unpickle and return object located at filename.
    """

    with open(filename, 'rb') as f:
        obj = pickle.load(f)

    return obj

def create_data_dir(root_dir_name):
    """
    Create a data directory stamped with the current time inside root_dir
    and return the path name.
    """

    dir_path_name = os.path.join(root_dir_name,
            datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.mkdir(dir_path_name)

    return dir_path_name


if __name__ == '__main__':

    data_dir = args.output_dir
    dqn_trainer_config = args.config
    num_episodes = args.num_episodes
    episode_length = args.episode_length
    load_checkpoint = args.load_checkpoint
    continue_training = args.continue_training


    env_tuples = get_env_tuples()

    dqn_trainer = trainer.DQNTrainer(dqn_trainer_config)
    if load_checkpoint is not None:
        dqn_trainer.load_checkpoint(load_checkpoint)

    for i, env_tuple in enumerate(env_tuples):

        print(f'Training run {i} commencing...')
        dqn_trainer.train(num_episodes, episode_length, env_tuple)

        print(f'Training run {i} completed, saving checkpoint...')
        data_dir_name = create_data_dir(data_dir)
        pickle_obj(os.path.join(data_dir_name, 'env_tuple.pkl'),
                   env_tuple)
        copyfile(dqn_trainer_config,
                 os.path.join(data_dir_name, 'config.yml'))
        dqn_trainer.save_checkpoint(
            os.path.join(data_dir_name, checkpoint_filename))
        print('Preparing plot...')
        dqn_trainer.plot(episode_length,
                         os.path.join(data_dir_name, plot_filename))
        print('Plotting complete.')
