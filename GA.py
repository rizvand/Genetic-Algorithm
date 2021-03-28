import numpy as np
from testfunction import ackley, sphere

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
            point = int(np.random.randint(self.dim, size = 1))
            self.values[:point] = maternal_chromosome.values[:point].copy()
            
        elif method == 'two-point':
            r1, r2 = np.random.randint(self.dim, size = 2)
            point1 = min(r1,r2)
            point2 = max(r1,r2)
            
            self.values[:point1] = maternal_chromosome.values[:point1].copy()
            self.values[point2:] = maternal_chromosome.values[point2:].copy()
        
        #fix this
        elif method == 'uniform':
            r = np.random.randint(2, size = self.dim)
            for i in range(self.dim):
                if r[i] == 1:
                    self.values[i] = maternal_chromosome.values[i].copy()
        return self
    
class Population():
    def __init__(self, random_state=1):
        self.best_chromosome = None
        self.best_fitness = None
        np.random.seed(random_state)
        
    def initialize(self, size, boundary):
        population = []
        for i in range(size):
            values = []
            for b in boundary:
                if b[2] == 'int':
                    val = np.random.randint(b[0], b[1]+1)
                elif b[2] == 'float':
                    val = np.random.uniform(b[0], b[1]) 
                values.append(val)
            initial_chromosome = Chromosome(values, random_state = i)
            population.append(initial_chromosome)
        self.history_population = population
        self.pop_size = size
    
    def get_fitness(self, f_obj):
        self.history_fitness = []
        for x in self.history_population:
            x.fit(f_obj)
            self.history_fitness.append(x.fitness)
    
    def get_best(self, objective='minimum'):
        if objective == 'minimum':
            idx_best = np.argmin(self.history_fitness)
        elif objective == 'maximum':
            idx_best = np.argmax(self.history_fitness)
        self.best_fitness = self.history_fitness[idx_best]
        self.best_chromosome = self.history_population[idx_best]
    
    def selection(self, method):
        if method == 'roulette':
            #return selection proba for each chromosome (list)
            self.selection_method = 'roulette'
            sum_of_fitness = sum(self.history_fitness)
            try:
                if sum_of_fitness != 0:
                    self.roulette_proba = np.array(self.history_fitness)/(sum_of_fitness)
                else:
                    print('masuk')
                    self.roulette_proba = np.ones(self.pop_size)/self.pop_size
            except:
                self.roulette_proba = np.ones(self.pop_size)/self.pop_size
                
        elif method == 'tournament':
            #fixed tournament size = half of total population
            self.selection_method == 'tournament'
            pairs = []
            tournament_size = round(self.pop_size/4)
            for i in range(pop_size):
                pool_idx = np.random.randint(0, len(self.history_population), size=tournament_size)
                pool_fitness = np.array(self.history_fitness)[[pool_idx]]
                best_two = pool_fitness.argsort()[:2]
                idx = pool_idx[best_two]
                couple = self.history_population[idx[0]], self.history_population[idx[1]]
                pairs.append(couple)
            
    def crossover(self, method):
        if self.selection_method == 'roulette':
            self.offspring_population = []
            cdf_proba = np.array([self.roulette_proba[:i+1].sum() for i in range(self.roulette_proba.shape[0])])
            for i in range(self.pop_size):
                r1 = np.random.rand()
                r2 = np.random.rand()
                
                idx1 = np.min(np.where(r1 <= cdf_proba))
                idx2 = np.min(np.where(r2 <= cdf_proba))
                    
                paternal = self.history_population[idx1]
                maternal = self.history_population[idx2]
                offspring = paternal.crossover(maternal, method=method)
                self.offspring_population.append(offspring)
                
        elif self.selection_method == 'tournament':
            self.offspring_population = []
            for pair in pairs:
                paternal = pair[0]
                maternal = pari[1]
                offspring = paternal.crossover(maternal, method=method)
                self.offspring_population.append(offspring)
              
    def mutation(self, method, prob, boundary):
        def cut(x, bound):
            if x < bound[0]:
                x = bound[0]
            elif x > bound[1]:
                x = bound[1]
            if bound[2] == 'float':
                x = float(x)
            elif bound[2] == 'int':
                x = int(x)
            return x
        
        for offspring in self.offspring_population:
            offspring.mutate(method=method, mutation_prob=prob)
            for i, bound in enumerate(boundary):
                cut(offspring.values[i], bound)
    
    def replacement(self, method, frac, obj_func):
        if method == 'partial':
            num_keep = int(frac*self.pop_size)
            offspring_fitness = np.array([obj_func(x.values) for x in self.offspring_population])
            idx_offspring_keep = offspring_fitness.argsort()[:num_keep]
            offspring_keep = np.array(self.offspring_population)[idx_offspring_keep]
            idx_keep = np.array(self.history_fitness).argsort()[:num_keep]
            pop_keep = np.array(self.history_population)[idx_keep]
            self.history_population = np.append(pop_keep, offspring_keep).tolist()

class GeneticAlgorithm:
    def __init__(self, size, selection, crossover, mutation, replacement, crossover_prob, mutation_prob, replacement_frac, n_iter, boundary, objective_function, objective, random_state, save_history):
        self.size = size
        self.n_iter = n_iter
        self.boundary = boundary
        self.objective_function = objective_function
        self.objective = objective
        self.selection_method = selection
        self.crossover_method = crossover
        self.mutation_method = mutation
        self.replacement_method = replacement
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.replacement_frac = replacement_frac
        self.random_state = random_state
        self.save_history = save_history
    
    def simulate(self):
        population = Population(random_state = self.random_state)
        population.initialize(self.size, self.boundary)
        population.get_fitness(self.objective_function)
        population.get_best(objective='minimum')
        
        if self.save_history:
            self.history = []
            self.history.append(population.history_population)
            
        iteration = 0
        print('iteration: ', iteration, ' | best chromosome: ', population.best_chromosome.values, ' | best fitness: ', population.best_fitness)
        
        while iteration < self.n_iter:
            iteration += 1
            population.selection(method=self.selection_method)
            population.crossover(method=self.crossover_method)
            population.mutation(method=self.mutation_method, prob=self.mutation_prob, boundary=self.boundary)
            population.replacement(method=self.replacement_method, frac=self.replacement_frac, obj_func=self.objective_function)
            population.get_fitness(self.objective_function)
            population.get_best(objective='minimum')
            if self.save_history:
                self.history.append(population.history_population)
            print('iteration: ', iteration, ' | best chromosome: ', population.best_chromosome.values, ' | best fitness: ', population.best_fitness)