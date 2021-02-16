import time

import src.dqn.trainer as trainer
from src.instance_parser import _unzip_instances, unzip_sort_parse


def instances_to_tuples():
    """
    Returns a tuple (starts, targets, obstacles, instance) for each instance
    in official_instances.zip.
    """
    return unzip_sort_parse()

# Solution zip filename
solution_zip_filename = 'solution.zip'

# DQN agent information
checkpoint_filename = 'checkpoint.pt'
config_filename = 'config.yml'

# Solution parameters
max_solution_timesteps = 1000


if __name__ == '__main__':

    # get the instances
    instances = _unzip_instances()
    env_tuples = unzip_sort_parse() # instances_to_tuples(instances)

    # import pdb; pdb.set_trace()

    dqn_solver = trainer.DQNTrainer(config_filename)
    dqn_solver.load_checkpoint(checkpoint_filename, continue_training=False)

    for i, elem in enumerate(env_tuples):
        print(f'Instance {i} run commencing...')
        t0 = time.time()
        dqn_solver.solve(episode_length=max_solution_timesteps, env_tuple=elem)
        t1 = time.time()

        print(f'Run completed after {t1 - t0}s.')
        print('Writing solution...')
        dqn_solver.write_solution(solution_zip_filename)
        print('Writing complete.')
