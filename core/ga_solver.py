import random
import time
import copy
from genetic.mutation import Mutation
from genetic.crossover import Crossover

class GASolver:

    def __init__(self, 
                solution_template, 
                population_size=150, 
                generations=1000, 
                tournament_size=3, 
                elitism_size=2, 
                time_limit=30,
                eps=1e-6,
                solution_distance_threshold=0.01):  
        
        self.solution_template = solution_template
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.elitism_size = elitism_size
        self.eps = eps
        self.solution_distance_threshold = solution_distance_threshold
        
        self.mutation = Mutation({
            'rate': 0.2,
            'method': 'gaussian',
            'scale': 0.1
        })
        
        self.crossover = Crossover({
            'method': 'arithmetic',
            'rate': 0.8,
            'alpha': 0.5
        })
        
        self.population = []
        self.fitness_values = []
        self.graduated_solutions = []  
        self.initialize_population()
        self.update_fitness_values()
        
        self.start_time = time.process_time()
        self.end_time = self.start_time + time_limit

    def initialize_population(self):
        for _ in range(self.population_size):
            new_solution = copy.deepcopy(self.solution_template)
            
            for var_name, variable in new_solution.variables.items():
                domain_min = variable.domain_min
                domain_max = variable.domain_max
                
                if domain_min == float('-inf'):
                    domain_min = -100
                if domain_max == float('inf'):
                    domain_max = 100
                
                random_value = random.uniform(domain_min, domain_max)
                variable.set_value(random_value)
            
            self.population.append(new_solution)

    def update_fitness_values(self):
        self.fitness_values = []
        for solution in self.population:
            heuristic = solution.calculate_heuristic()
            if heuristic is None:
                heuristic = float('inf')
            self.fitness_values.append(heuristic)

    def tournament_selection(self):
        tournament = random.sample(list(zip(self.population, self.fitness_values)), 
                                  self.tournament_size)
        return min(tournament, key=lambda x: x[1])[0]

    def add_random_individuals(self, num_random_individuals):
        sorted_indices = sorted(range(len(self.fitness_values)), 
                               key=lambda i: self.fitness_values[i])
        self.population = [self.population[i] for i in sorted_indices[:len(self.population) - num_random_individuals]]
        
        for _ in range(num_random_individuals):
            new_solution = copy.deepcopy(self.solution_template)
            for var_name, variable in new_solution.variables.items():
                domain_min = variable.domain_min if variable.domain_min != float('-inf') else -100
                domain_max = variable.domain_max if variable.domain_max != float('inf') else 100
                random_value = random.uniform(domain_min, domain_max)
                variable.set_value(random_value)
            self.population.append(new_solution)

    def ga_step(self):
        new_population = []
        
        for _ in range(self.population_size // 2):  
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            child1, child2 = self.crossover.forward(parent1, parent2)
            
            child1 = self.mutation.forward(child1)
            child2 = self.mutation.forward(child2)
            
            new_population.extend([child1, child2])
        
        combined_population = self.population + new_population
        combined_fitness = self.fitness_values + [
            solution.calculate_heuristic() or float('inf') 
            for solution in new_population
        ]
        
        sorted_indices = sorted(range(len(combined_fitness)), 
                               key=lambda i: combined_fitness[i])
        
        self.population = [combined_population[i] for i in 
                         sorted_indices[:self.population_size]]
        
        self.update_fitness_values()
        
        best_idx = self.fitness_values.index(min(self.fitness_values))
        return self.population[best_idx]

    def solution_distance(self, solution1, solution2):
        sum_squared_diff = 0
        for var_name in solution1.variables:
            if var_name in solution2.variables:
                val1 = solution1.variables[var_name].current_value
                val2 = solution2.variables[var_name].current_value
                if val1 is not None and val2 is not None:
                    sum_squared_diff += (val1 - val2) ** 2
        
        return sum_squared_diff ** 0.5

    def is_new_solution(self, solution):
        for found_solution in self.graduated_solutions:
            if self.solution_distance(solution, found_solution) < self.solution_distance_threshold:
                return False
        print("New solution found!")
        # Print out solution details
        for var_name, variable in solution.variables.items():
            print(f"{var_name} = {variable.current_value:.6f}")
        print(f"Error: {solution.calculate_heuristic()}")
        return True

    def graduate_solution(self, solution, fitness):
        if not self.is_new_solution(solution):
            return 0  
            
        self.graduated_solutions.append(copy.deepcopy(solution))
        
        to_remove = []
        for i, pop_solution in enumerate(self.population):
            if self.solution_distance(solution, pop_solution) < self.solution_distance_threshold:
                to_remove.append(i)
        
        for idx in sorted(to_remove, reverse=True):
            if idx < len(self.population):
                del self.population[idx]
                del self.fitness_values[idx]
        
        return len(to_remove)

    def solve(self):
        stable_count = 0
        
        while time.process_time() <= self.end_time and len(self.population) > self.population_size // 2:
            solution = self.ga_step()
            fitness = solution.calculate_heuristic()
            
            if fitness is None:
                continue
                
            if fitness < self.eps:
                removed_count = self.graduate_solution(solution, fitness)
                if removed_count > 0:
                    self.add_random_individuals(removed_count)
                    stable_count = 0
                    self.update_fitness_values()
                    continue            
            
            stable_count += 1
            
            if stable_count >= 20:
                self.mutation.rate = min(0.5, self.mutation.rate * 1.5)  
            if stable_count >= 50:
                self.mutation.rate = min(0.8, self.mutation.rate * 1.2) 
            if stable_count >= 100:
                self.add_random_individuals(self.population_size // 2)
                self.mutation.rate = 0.2  
                stable_count = 0
                
            if stable_count % 10 == 0 and stable_count > 0:
                self.add_random_individuals(max(5, self.population_size // 20))
        
        if not self.graduated_solutions and self.population:
            best_idx = self.fitness_values.index(min(self.fitness_values))
            self.graduated_solutions.append(self.population[best_idx])
                
        return self.graduated_solutions

    def set_mutation_config(self, config):
        self.mutation = Mutation(config)
        
    def set_crossover_config(self, config):
        self.crossover = Crossover(config)