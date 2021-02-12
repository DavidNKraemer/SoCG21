import argparse
import os
from datetime import datetime
from pathlib import Path
import sys
from functools import partial
import pdb


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.genetic.utils import NogradModule
from src.genetic.genetic_algorithm import BoardGA
from src.board import LocalState

from training_sequence import training_sequence, training_plan


# Argument parsing setup

today = datetime.today()
timestr = today.strftime('%Y_%m_%d_%H%M%S')
default_dir = os.path.join(
    'experiments','genetic_algorithm', f'ga_experiment_{timestr}'
)

parser = argparse.ArgumentParser(
    description='Main genetic algorithm experiment script'
)

parser.add_argument(      '--out_dir', type=str,   default=default_dir)
parser.add_argument( '--n_population', type=int,   default=100)
parser.add_argument(    '--n_parents', type=int,   default=20)
parser.add_argument('--n_generations', type=int,   default=6)
parser.add_argument(    '--max_clock', type=int,   default=50)
parser.add_argument(    '--len_epoch', type=int,   default=10)
parser.add_argument(       '--myopia', type=float, default=0.9)


# Policy model factory
def model_builder(in_features, out_features):
    width = 64
    model = NogradModule(nn.Sequential(
        nn.Linear(in_features, width),
        nn.ReLU(),
        nn.Linear(width, width),
        nn.ReLU(),
        nn.Linear(width, out_features),
        nn.Softmax(dim=0)
    ))
    model.values = torch.zeros(model.size)

    return model

builder = partial(model_builder, LocalState.shape[0], 5)



if __name__ == '__main__':
    args = parser.parse_args()

    kwargs = {
        'n_population': args.n_population,
        'n_parents': args.n_parents,
        'max_clock': args.max_clock,
        'n_generations': args.n_generations,
        'len_epoch': args.len_epoch,
        'myopia': args.myopia,
    }

    # set up output directory
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    ga = BoardGA(builder, **kwargs)

    plan = training_plan(training_sequence, kwargs['myopia'], kwargs['len_epoch'])

    for i, (description, sources, targets, obstacles) in enumerate(plan):
        print(f"Problem {i}: {description}.", end=" ")

        ga.set_env(sources, targets, obstacles, max_clock=kwargs['max_clock']) 
        ga.train(kwargs['n_generations'])

        # evaluate the last generation
        ga.evaluate()  
        pdb.set_trace()

        best = ga.optimal_policy()
        print(f"fitness: {best.fitness}.", end=" ")
        print("saving...", end=" ")
        best.save_model(os.path.join(args.out_dir, f'problem_{i}.model'))
        print("done!")
