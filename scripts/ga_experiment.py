import argparse
import os
from datetime import datetime
from pathlib import Path
import sys


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.genetic.utils import NogradModule
from src.genetic.genetic_algorithm import BoardGA
from src.board import LocalState

from training_sequence import training_sequence


# Argument parsing setup

today = datetime.today()
timestr = today.strftime('%Y_%m_%d_%H%M%S')
default_dir = os.path.join(
    'experiments','genetic_algorithm', f'ga_experiment_{timestr}'
)

parser = argparse.ArgumentParser(
    description='Main genetic algorithm experiment script'
)

parser.add_argument('--out_dir', type=str, default=default_dir)
parser.add_argument('--n_population', type=int, default=200)
parser.add_argument('--n_parents', type=int, default=10)
parser.add_argument('--n_generations', type=int, default=1000)
parser.add_argument('--max_clock', type=int, default=30)


# Policy model factory
def model_builder(in_features, out_features):
    width = 64
    def builder():
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
    return builder

builder = model_builder(LocalState.shape[0], 5)



if __name__ == '__main__':
    args = parser.parse_args()

    kwargs = {
        'n_population': args.n_population,
        'n_parents': args.n_parents,
        'max_clock': args.max_clock,
        'n_generations': args.n_generations
    }

    # set up output directory
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    ga = BoardGA(builder, **kwargs)

    for i, training_dict in enumerate(training_sequence):
        sources = training_dict['sources']
        targets = training_dict['targets']
        obstacles = training_dict['obstacles']

        print(f"Problem {i}: {training_dict['description']}")

        ga.set_env(sources, targets, obstacles, max_clock=kwargs['max_clock']) 

        ga.train(kwargs['n_generations'])

        best = ga.optimal_policy()
        print("saving...", end=" ")
        best.save(args.out_dir)
        print("done!")
