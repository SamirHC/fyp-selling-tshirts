import numpy as np


class MAPElites:
    def __init__(self, bins, fitness_func, mutation_func):
        self.bins = bins
        self.mutation_func = mutation_func
        self.fitness_func = fitness_func

        self.archive = {}
    
    def evolve(self):
        pass
