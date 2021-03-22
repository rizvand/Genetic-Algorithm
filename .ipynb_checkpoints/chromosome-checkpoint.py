import numpy as np

class Chromosome():
    def __init__(self, values, random_state=1):
        self.values = np.array(values)
        self.dim = len(values)
        np.random.seed(random_state)
    
    def fit(self, f_obj):
        self.fitness = f_obj(self.values)
    
    def mutate(self, method='swap', mutation_prob=0.5):
        if method == 'swap':
            idx1, idx2 = np.random.randint(self.dim, size=2)
            self.values[[idx1, idx2]] = self.values[[idx2, idx1]]
    
    def crossover(self, maternal_chromosome, method='one-point'):
        if method == 'one-point':
            point = np.random.randint(self.dim, size = 1)
            self.values[:point] = maternal_chromosome.values[:point].copy()
            
        elif method == 'two-point':
            r1, r2 = np.random.randint(self.dim, size = 2)
            point1 = min(r1,r2)
            point2 = max(r1,r2)
            
            self.values[:point1] = maternal_chromosome.values[:point1].copy()
            self.values[point2:] = maternal_chromosome.values[point2:].copy()
            
        elif method == 'uniform':
            r = np.random.randint(2, size = self.dim)
            for i in range(self.dim):
                if r[i] == 1:
                    self.values[i] = maternal_chromosome.values[i].copy()