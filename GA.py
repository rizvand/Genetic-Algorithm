import numpy as np
from numpy.random import default_rng
from testfunction import ackley, sphere
import copy

class Chromosome:
    def __init__(self, values, boundary):
        self.values = np.array(values)
        self.boundary = boundary
        self.dim = len(values)
    
    def fit(self, fitness_function):
        self.fitness = fitness_function(self.values)
    
    def mutation(self, method, mutation_prob, random_state):
        generator = default_rng(seed=random_state)
        if method == 'swap':
            idx1, idx2 = generator.choice(self.dim, size=2)
            self.values[[idx1, idx2]] = self.values[[idx2, idx1]]
    
    def penalty(self):
        for i in range(self.dim):
            if self.values[i] > self.boundary[i][1]:
                self.values[i] == self.boundary[i][0]
            elif self.values[i] < self.boundary[i][0]:
                self.values[i] == self.boundary[i][0]

class GeneticAlgorithm:
    def __init__(self, size, n_iter, fitness_function, boundary, selection, crossover, mutation, elitism, crossover_prob, mutation_prob, save_history, random_state):
        self.size = size
        self.n_iter = n_iter
        self.fitness_function = fitness_function
        self.boundary = boundary
        self.selection_method = selection
        self.crossover_method = crossover
        self.mutation_method = mutation
        self.elitism = elitism
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.save_history = save_history
        self.rng = default_rng(seed=random_state)
        self.dim = len(boundary)
        np.random.seed(random_state)
        
    def roulette_selection(self, population):
        population_fitness = np.array([x.fitness for x in population])
        sum_of_fitness = population_fitness.sum()
        if sum_of_fitness != 0:
            roulette_proba = (sum_of_fitness-population_fitness)/(sum_of_fitness)
            cdf_proba = np.array([roulette_proba[:i+1].sum() for i in range(roulette_proba.shape[0])])
            r1 = np.random.rand()
            r2 = np.random.rand()
            idx1 = np.min(np.where(r1 <= cdf_proba))
            idx2 = np.min(np.where(r2 <= cdf_proba))
            paternal = copy.deepcopy(population[idx1])
            maternal = copy.deepcopy(population[idx2])
        else:
            paternal = copy.deepcopy(population[0])
            maternal = copy.deepcopy(population[1])
        return paternal, maternal

    def tournament_selection(self, population):
        tournament_size = round(self.size/4)
        tournament_idx = self.rng.choice(self.size, size=tournament_size, replace=False)
        tournament_pool = copy.deepcopy(np.array(population)[tournament_idx])
        tournament_fitness = [x.fitness for x in tournament_pool]
        best_idx = np.array(tournament_fitness).argsort()[:2]
        paternal, maternal = copy.deepcopy(tournament_pool[best_idx[0]]), copy.deepcopy(tournament_pool[best_idx[1]]) 
        return paternal, maternal

    def pair_selection(self, population):
        if self.selection_method == 'roulette':
            paternal, maternal = self.roulette_selection(population)
            
        elif self.selection_method == 'tournament':
            paternal, maternal = self.tournament_selection(population)
            
        return paternal, maternal

    def crossover(self, paternal_chromosome, maternal_chromosome):
        if self.crossover_method == 'one-point':
            random_point = np.random.randint(self.dim)
            offspring = copy.deepcopy(paternal_chromosome)
            offspring.values[random_point:] = copy.deepcopy(maternal_chromosome.values[random_point:])
            return offspring

        elif self.crossover_method == 'two-point':
            random_point = np.sort(self.rng.choice(self.dim, size = 2))
            offspring = copy.deepcopy(paternal_chromosome)
            offspring.values[random_point[0]:random_point[1]] = copy.deepcopy(maternal_chromosome.values[random_point[0]:random_point[1]])
            return offspring
        
        elif self.crossover_method == 'uniform':
            offspring = copy.deepcopy(paternal_chromosome)
            for i in range(self.dim):
                r = np.random.rand()
                if r > self.crossover_prob:
                    offspring.values[i] = copy.deepcopy(maternal_chromosome.values[i])
            return offspring

    def simulate(self):
        # Initialize
        population = []
        for i in range(self.size):
            position = []
            for pos_bound in self.boundary:
                pos_ = np.random.uniform(pos_bound[0], pos_bound[1])
                position.append(pos_)
            initial_chromosome = Chromosome(position, self.boundary)
            initial_chromosome.fit(self.fitness_function)
            population.append(initial_chromosome)

        iteration_fitness = np.array([x.fitness for x in population])
        best_idx = np.argmin(iteration_fitness)
        iteration_best = population[best_idx].values, population[best_idx].fitness

        if self.save_history == True:
            self.population_history = []
            self.fitness_history = []
            self.best_history = []
            self.population_history.append([x.values for x in population.copy()])
            self.fitness_history.append(iteration_fitness.copy())
            self.best_history.append(iteration_best)
        
        iteration = 0
#         print(f'iteration: {iteration} | best_position : {iteration_best[0]} | best_fit : {iteration_best[1]}')

        #Optimization
        while iteration < self.n_iter:
            iteration += 1
            offspring_population = []
            for j in range(self.size):
                paternal, maternal = self.pair_selection(population)
                offspring = self.crossover(paternal, maternal)
                offspring.mutation(method=self.mutation_method, mutation_prob=self.mutation_prob, random_state=j*iteration)
                offspring.penalty()
                offspring.fit(self.fitness_function)
                offspring_population.append(offspring)

            # get (1 - elitism) offspring chromosome
            offspring_fitness = np.array([x.fitness for x in offspring_population])
            num_keep = int((1 - self.elitism)*self.size)
            keep_idx = offspring_fitness.argsort()[:num_keep]
            selected_offspring = np.array(offspring_population)[keep_idx]

            # get elitism previous generation chromosome
            population_fitness = np.array([x.fitness for x in population])
            num_keep = int(self.elitism*self.size)
            keep_idx = population_fitness.argsort()[:num_keep]
            selected_population = np.array(population)[keep_idx]

            new_population = np.concatenate((selected_population, selected_offspring))
            iteration_fitness = np.array([x.fitness for x in new_population])
            best_idx = np.argmin(iteration_fitness)
            iteration_best = new_population[best_idx].values, new_population[best_idx].fitness
#             print(f'iteration: {iteration} | best_position : {iteration_best[0]} | best_fit : {iteration_best[1]}')
            population = new_population.copy()

            if self.save_history == True:
                self.population_history.append([x.values for x in population])
                self.fitness_history.append(iteration_fitness)
                self.best_history.append(iteration_best)

            
        

