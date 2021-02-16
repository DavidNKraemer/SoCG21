import time

from cgshop2021_pyutils import SolutionZipWriter
import src.dqn.trainer as trainer
from src.instance_parser import _unzip_instances, unzip_sort_parse

# Solution zip filename
solution_zip_filename = 'solution.zip'

# DQN agent information
checkpoint_filename = 'checkpoint.pt'
config_filename = 'config.yml'

# Solution parameters
max_solution_timesteps = 500


if __name__ == '__main__':

    # get the instances
    env_tuples = unzip_sort_parse()[:2]

    dqn_solver = trainer.DQNTrainer(config_filename)
    dqn_solver.load_checkpoint(checkpoint_filename, continue_training=False)

    solutions = []

    for i, elem in enumerate(env_tuples):
        print(f'Instance {i} run commencing...')
        t0 = time.time()
        dqn_solver.solve(episode_length=max_solution_timesteps, env_tuple=elem)
        t1 = time.time()

        print(f'Run completed after {t1 - t0:.2f}s.')
        print('Writing solution...')
        solutions.append(dqn_solver.get_solution())
        print('Writing complete.')

    # write solutions to zipfile
    with SolutionZipWriter(solution_zip_filename) as szw:
        szw.add_solutions(solutions)
