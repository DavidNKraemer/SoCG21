import torch
from abc import ABC, abstractmethod
import random
from operator import methodcaller


class GeneticAlgorithm(ABC):
    """
    Abstract base class for all genetic algorithms.
    """

    @abstractmethod
    def initialize(self):
        """
        Initializes the population of the first generation of the genetic
        algorithm.
        """
        pass

    @abstractmethod
    def evaluate(self):
        """
        Evaluates the performance of members of the current generation against
        a fitness function.
        """
        pass

    @abstractmethod
    def select(self):
        """
        Selects a subset of the current generation for breeding in the next
        generation.
        """
        pass

    @abstractmethod
    def cross(self):
        """
        Performs breeding of the selected parents in the current generation,
        producing the next generation.
        """
        pass

    @abstractmethod
    def mutate(self):
        """
        The values of the next generation are subject to mutation before
        starting again
        """
        pass


class Crossover:
    __instance = None
    def __new__(cls, *args):
        if cls.__instance is None:
            cls.__instance = object.__new__(cls, *args)
        return cls.__instance

    @staticmethod
    def amxo(p1, p2):
        """
        Given parent parameters p1 and p2, compute the crossover parameter x
        defined by

        x[i] = alpha[i] * p1[i] + (1. - alpha[i]) * p2[i]

        where alpha is a uniform random variable of identical shape
        """
        assert p1.shape == p2.shape

        alpha = torch.rand(*p1.shape)
        return alpha * p1 + (1. - alpha) * p2

    @staticmethod
    def heuristic(p1, p2):
        """
        Given parent parameters p1 and p2, compute the crossover parameter x
        defined by

        x[i] = alpha * (p2[i] - p1[i]) + p2[i]

        where alpha is a uniform random variable in [0,1]
        """
        assert p1.shape == p2.shape

        alpha = random.random()
        return alpha * (p2 - p1) + p2

    @staticmethod
    def laplace(p1, p2, loc=0., scale=1.):
        assert p1.shape == p2.shape

        alpha = torch.rand(1)
        coeff = 2. * (alpha <= 0.5) - 1.
        beta = loc - coeff * scale * torch.log(alpha)

        x1, x2 = p1 + beta * torch.abs(p1 - p2), p2 + beta * torch.abs(p1 - p2)
        return x1 if random.random() < 0.5 else x2

    @staticmethod
    def blxa(p1, p2, alpha=0.5):
        assert p1.shape == p2.shape

        upper = torch.max(p1, p2)
        lower = torch.min(p1, p2)
        length = upper - lower
        upper += length * alpha
        lower -= length * alpha

        return lower + (upper - lower) * torch.rand(p1.shape)

    @staticmethod
    def pbxa(p1, p2, alpha=0.5):
        assert p1.shape == p2.shape

        upper = torch.max(p1, p2)
        lower = torch.min(p1, p2)
        length = upper - lower
        upper += length * alpha
        lower -= length * alpha

        return lower + (upper - lower) * torch.rand(p1.shape)


crossover = Crossover()


class DemoGA(GeneticAlgorithm):
    """
    Use the GeneticAlgorithm base class to perform a line of best fit.
    """
    class Specie:
        def __init__(self, weights):
            self.weights = weights

        def fitness(self, inputs, outputs):
            """
            Total absolute error
            """
            offset = torch.ones(inputs.shape[0], 1)
            X = torch.cat((offset, inputs), dim=1)
            y = outputs.reshape(-1, len(outputs))

            pred = X @ self.weights
            error = ((torch.abs(y - pred)) ** 2).mean()

            return -float(error)

        def __repr__(self):
            return str(self.weights)

    def __init__(self, inputs, outputs, n_population=1000, n_parents=100):
        """
        Params
        ------
        inputs: torch.tensor
            Input data
        outputs: torch.tensor
            True outputs
        n_population: int
            Size of the population in each generation
        n_parents: int
            Number of parents to take from the population for crossbreeding
        """
        self.inputs = inputs
        self.outputs = outputs
        self.n_population = n_population
        self.n_parents = n_parents

        self.weight_shape = self.inputs.shape[1] + 1
        self.evaluator = methodcaller('fitness', self.inputs, self.outputs)

    def initialize(self):
        """
        Starts the first generation with a new population
        """
        self.generation = 1
        self.population = [
            self.Specie(torch.randn(self.weight_shape)) \
            for _ in range(self.n_population)
        ]

    def evaluate(self):
        """
        Sorts the population by fitness scores
        """
        assert self.__getattribute__('population'), "Run initialize first!"

        self.population.sort(key=self.evaluator, reverse=True)

    def select(self):
        """
        Identifies the parents of the generation by the highest scoring members
        """
        assert self.__getattribute__('population'), "Run evaluate first!"

        self.parents = self.population[:self.n_parents]

    def cross(self):
        """
        By randomly selecting parents to breed, breed a new population from the
        parents
        """
        assert self.__getattribute__('parents'), "Run select first"

        self.population = []
        for _ in range(self.n_population):
            p1, p2 = random.choices(self.parents, k=2)
            weights = crossover.pbxa(p1.weights, p2.weights)
            self.population.append(self.Specie(weights))
        self.generation += 1

    def mutate(self):
        """
        Modify the genomes of the current population
        """
        assert self.__getattribute__('population'), "Run initialize first!"

        for specie in self.population:
            specie.weights += torch.randn(specie.weights.shape)

