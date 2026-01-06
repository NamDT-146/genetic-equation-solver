import random
import time
import copy
from queue import Queue
import numpy as np
from .genetic.mutation import Mutation
from .genetic.crossover import Crossover

class GASolver:

    def __init__(self, 
                solution_template, 
                population_size=150, 
                generations=1000, 
                tournament_size=3, 
                elitism_size=2, 
                time_limit=30,
                eps=1e-6,
                solution_distance_threshold=0.01,
                visualization_queue=None,
                visualization_interval=5,
                num_islands=4,
                migration_interval=10,
                migration_rate=0.1):  
        
        self.solution_template = solution_template
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.elitism_size = elitism_size
        self.eps = eps
        self.solution_distance_threshold = solution_distance_threshold
        
        self.num_islands = num_islands
        self.migration_interval = migration_interval
        self.migration_rate = migration_rate
        self.island_size = population_size // num_islands
        
        self.visualization_queue = visualization_queue
        self.visualization_interval = visualization_interval
        self.generation_count = 0
        
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.worst_fitness_history = []
        
        self.islands = [] 
        self.island_fitness = []  
        self.island_mutations = []  
        self.island_crossovers = []  
        
        self.graduated_solutions = []  
        
        self.initialize_islands()
        
        self.start_time = time.time()
        self.end_time = self.start_time + time_limit
        self.time_limit = time_limit

    def initialize_islands(self):
        mutation_configs = [
            {'rate': 0.1, 'method': 'gaussian', 'scale': 0.05}, 
            {'rate': 0.2, 'method': 'gaussian', 'scale': 0.1},   
            {'rate': 0.3, 'method': 'uniform', 'scale': 0.15},  
            {'rate': 0.4, 'method': 'gaussian', 'scale': 0.2}, 
        ]
        
        crossover_configs = [
            {'method': 'arithmetic', 'rate': 0.9, 'alpha': 0.5},
            {'method': 'single_point', 'rate': 0.8, 'alpha': 0.5},
            {'method': 'arithmetic', 'rate': 0.7, 'alpha': 0.3},
            {'method': 'arithmetic', 'rate': 0.85, 'alpha': 0.7},
        ]
        
        for island_idx in range(self.num_islands):
            mutation_config = mutation_configs[island_idx % len(mutation_configs)]
            crossover_config = crossover_configs[island_idx % len(crossover_configs)]
            
            self.island_mutations.append(Mutation(mutation_config))
            self.island_crossovers.append(Crossover(crossover_config))
            
            island_population = []
            for _ in range(self.island_size):
                new_solution = copy.deepcopy(self.solution_template)
                
                for var_name, variable in new_solution.variables.items():
                    domain_min = variable.domain_min
                    domain_max = variable.domain_max
                    random_value = random.uniform(domain_min, domain_max)
                    variable.set_value(random_value)
                
                island_population.append(self.local_search(new_solution))
            
            self.islands.append(island_population)
            
            island_fitness = []
            for solution in island_population:
                heuristic = solution.calculate_heuristic()
                if heuristic is None:
                    heuristic = float('inf')
                island_fitness.append(heuristic)
            
            self.island_fitness.append(island_fitness)
        
        self.update_global_fitness_history()

    @property
    def population(self):
        flat_population = []
        for island in self.islands:
            flat_population.extend(island)
        return flat_population

    @property
    def fitness_values(self):
        flat_fitness = []
        for island_fit in self.island_fitness:
            flat_fitness.extend(island_fit)
        return flat_fitness

    def update_global_fitness_history(self):
        all_fitness = []
        for island_fit in self.island_fitness:
            all_fitness.extend([f for f in island_fit if f != float('inf')])
        
        if all_fitness:
            self.best_fitness_history.append(min(all_fitness))
            self.avg_fitness_history.append(sum(all_fitness) / len(all_fitness))
            self.worst_fitness_history.append(max(all_fitness))

    def get_progress(self):
        elapsed = time.time() - self.start_time
        return min(100, (elapsed / self.time_limit) * 100)

    def push_visualization_data(self):
        if self.visualization_queue is None:
            return
        
        var_data = {}
        var_names = list(self.solution_template.variables.keys())
        
        for var_name in var_names:
            var_data[var_name] = []
            for island in self.islands:
                for solution in island:
                    if var_name in solution.variables:
                        val = solution.variables[var_name].current_value
                        if val is not None:
                            var_data[var_name].append(val)
        
        data = {
            'generation': self.generation_count,
            'progress': self.get_progress(),
            'population_data': var_data,
            'fitness_values': self.fitness_values,
            'best_fitness_history': self.best_fitness_history.copy(),
            'avg_fitness_history': self.avg_fitness_history.copy(),
            'worst_fitness_history': self.worst_fitness_history.copy(),
            'graduated_count': len(self.graduated_solutions)
        }
        
        try:
            self.visualization_queue.put_nowait(data)
        except:
            pass  

    def tournament_selection(self, island_idx):
        island = self.islands[island_idx]
        island_fit = self.island_fitness[island_idx]
        
        tournament = random.sample(list(zip(island, island_fit)), 
                                  min(self.tournament_size, len(island)))
        return min(tournament, key=lambda x: x[1])[0]

    def migrate_between_islands(self):
        num_migrants = max(1, int(self.island_size * self.migration_rate))
        for island_idx in range(self.num_islands):
            next_island_idx = (island_idx + 1) % self.num_islands

            island_with_fitness = list(zip(self.islands[island_idx], self.island_fitness[island_idx]))
            sorted_individuals = sorted(island_with_fitness, key=lambda x: x[1])
            migrants = [ind[0] for ind in sorted_individuals[:min(num_migrants, len(sorted_individuals))]]

            target_island = self.islands[next_island_idx]
            target_fitness = self.island_fitness[next_island_idx]
            target_with_fitness = list(zip(target_island, target_fitness))
            sorted_target = sorted(target_with_fitness, key=lambda x: x[1], reverse=True)

            replace_count = min(len(migrants), len(sorted_target))
            for i in range(replace_count):
                worst_idx = target_island.index(sorted_target[i][0])
                target_island[worst_idx] = copy.deepcopy(migrants[i])
                target_fitness[worst_idx] = migrants[i].calculate_heuristic() or float('inf')

    def add_random_individuals(self, island_idx, num_random_individuals):
        island = self.islands[island_idx]
        island_fit = self.island_fitness[island_idx]
        
        sorted_indices = sorted(range(len(island_fit)), key=lambda i: island_fit[i])
        keep_count = len(island) - num_random_individuals
        
        self.islands[island_idx] = [island[i] for i in sorted_indices[:keep_count]]
        self.island_fitness[island_idx] = [island_fit[i] for i in sorted_indices[:keep_count]]
        
        # Add random individuals
        for _ in range(num_random_individuals):
            new_solution = copy.deepcopy(self.solution_template)
            for var_name, variable in new_solution.variables.items():
                domain_min = variable.domain_min if variable.domain_min != float('-inf') else -100
                domain_max = variable.domain_max if variable.domain_max != float('inf') else 100
                random_value = random.uniform(domain_min, domain_max)
                variable.set_value(random_value)
            
            refined_solution = self.local_search(new_solution)
            self.islands[island_idx].append(refined_solution)
            fitness = refined_solution.calculate_heuristic() or float('inf')
            self.island_fitness[island_idx].append(fitness)

    # ...existing code...

    def local_search(self, solution, max_iterations=10, step_size=0.01, min_improvement=1e-8):
        import math
        
        best_solution = copy.deepcopy(solution)
        best_fitness = solution.calculate_heuristic()
        
        if best_fitness is None or best_fitness == float('inf'):
            return solution
        
        current_step = step_size
        current_solution = copy.deepcopy(best_solution)
        current_fitness = best_fitness
        
        initial_temperature = best_fitness * 0.1 if best_fitness > 0 else 0.1
        temperature = initial_temperature
        
        for iteration in range(max_iterations):
            improved = False
            
            temperature = initial_temperature * (1 - iteration / max_iterations)
            
            var_names = list(current_solution.variables.keys())
            random.shuffle(var_names)
            
            for var_name in var_names:
                variable = current_solution.variables[var_name]
                
                if variable.current_value is None:
                    continue
                
                original_value = variable.current_value
                domain_min = variable.domain_min 
                domain_max = variable.domain_max 
                
                exploration_strategy = random.random()
                
                if exploration_strategy < 0.3:  
                    direction = random.choice([-1, 1])
                    random_step = current_step * random.uniform(0.5, 2.0)
                    test_value = original_value + direction * random_step
                    test_value = max(domain_min, min(domain_max, test_value))
                    
                    variable.set_value(test_value)
                    test_fitness = current_solution.calculate_heuristic()
                    
                    if test_fitness is None:
                        test_fitness = float('inf')
                    
                    if test_fitness < current_fitness:
                        current_fitness = test_fitness
                        improved = True
                        if test_fitness < best_fitness:
                            best_fitness = test_fitness
                            best_solution = copy.deepcopy(current_solution)
                    elif temperature > 0:
                        delta = test_fitness - current_fitness
                        acceptance_probability = math.exp(-delta / temperature)
                        
                        if random.random() < acceptance_probability:
                            current_fitness = test_fitness
                            improved = True
                        else:
                            variable.set_value(original_value)
                    else:
                        variable.set_value(original_value)
                        
                else:  
                    test_value_pos = min(original_value + current_step, domain_max)
                    variable.set_value(test_value_pos)
                    fitness_pos = current_solution.calculate_heuristic()
                    
                    test_value_neg = max(original_value - current_step, domain_min)
                    variable.set_value(test_value_neg)
                    fitness_neg = current_solution.calculate_heuristic()
                    
                    variable.set_value(original_value)
                    
                    if fitness_pos is None:
                        fitness_pos = float('inf')
                    if fitness_neg is None:
                        fitness_neg = float('inf')
                    
                    # Initialize new_value and new_fitness to defaults
                    new_value = original_value
                    new_fitness = current_fitness
                    
                    if fitness_pos < current_fitness and fitness_pos <= fitness_neg:
                        new_value = test_value_pos
                        new_fitness = fitness_pos
                        improved = True
                    elif fitness_neg < current_fitness:
                        new_value = test_value_neg
                        new_fitness = fitness_neg
                        improved = True
                    else:
                        if fitness_pos < fitness_neg:
                            gradient_direction = 1
                        else:
                            gradient_direction = -1
                        
                        if random.random() < 0.3: 
                            gradient_direction *= -1
                        
                        adaptive_step = current_step * random.uniform(0.3, 0.7)
                        for _ in range(3):  
                            test_value = original_value + gradient_direction * adaptive_step
                            test_value = max(domain_min, min(domain_max, test_value))
                            
                            variable.set_value(test_value)
                            test_fitness = current_solution.calculate_heuristic()
                            
                            if test_fitness is not None and test_fitness < current_fitness:
                                new_value = test_value
                                new_fitness = test_fitness
                                improved = True
                                break
                            elif test_fitness is not None and temperature > 0:
                                delta = test_fitness - current_fitness
                                acceptance_probability = math.exp(-delta / temperature)
                                
                                if random.random() < acceptance_probability:
                                    new_value = test_value
                                    new_fitness = test_fitness
                                    improved = True
                                    break
                            
                            adaptive_step *= 0.5
                        
                        # If still not improved after adaptive search, restore original
                        if not improved:
                            variable.set_value(original_value)
                            continue
                    
                    # Only apply new_value if we actually improved
                    if improved and new_value != original_value:
                        variable.set_value(new_value)
                        current_fitness = new_fitness
                        if new_fitness < best_fitness:
                            best_fitness = new_fitness
                            best_solution = copy.deepcopy(current_solution)
            
            if not improved:
                current_step *= 0.5
                if current_step < step_size * 0.01:  
                    break
            else:
                current_step = min(step_size, current_step * 1.2)
        
        return best_solution

    def island_ga_step(self, island_idx):
        """Perform one GA step on a specific island"""
        island = self.islands[island_idx]
        island_fit = self.island_fitness[island_idx]
        mutation = self.island_mutations[island_idx]
        crossover = self.island_crossovers[island_idx]
        
        new_population = []
        
        for _ in range(len(island) // 2):  
            parent1 = self.tournament_selection(island_idx)
            parent2 = self.tournament_selection(island_idx)
            
            child1, child2 = crossover.forward(parent1, parent2)
            
            child1 = mutation.forward(child1)
            child2 = mutation.forward(child2)
            
            new_population.extend([self.local_search(child1), self.local_search(child2)])
        
        combined_population = island + new_population
        combined_fitness = island_fit + [
            solution.calculate_heuristic() or float('inf') 
            for solution in new_population
        ]
        
        sorted_indices = sorted(range(len(combined_fitness)), 
                               key=lambda i: combined_fitness[i])
        
        self.islands[island_idx] = [combined_population[i] for i in 
                                    sorted_indices[:len(island)]]
        self.island_fitness[island_idx] = [combined_fitness[i] for i in 
                                           sorted_indices[:len(island)]]
        
        for sol in self.islands[island_idx]:
            for var in sol.variables.values():
                if var.current_value is not None:
                    var.current_value = max(var.domain_min, min(var.domain_max, var.current_value))

        if not self.island_fitness[island_idx]:
            return copy.deepcopy(self.solution_template)
            
        best_idx = self.island_fitness[island_idx].index(min(self.island_fitness[island_idx]))
        return self.islands[island_idx][best_idx]


    def compute_island_diversity(self, island):
        if len(island) < 2:
            return 0.0
        n = len(island)
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                total += self.solution_distance(island[i], island[j])
                count += 1
        return total / count if count > 0 else 0.0

    def force_island_diversity(self, island_idx, threshold):
        island = self.islands[island_idx]
        
        # Check if island is empty or too small
        if len(island) < 2:
            return
        
        diversity = self.compute_island_diversity(island)
        max_attempts = 5
        attempts = 0
        while diversity < threshold and attempts < max_attempts:
            num_reinit = max(1, int(0.3 * len(island)))
            for _ in range(num_reinit):
                if len(island) == 0:
                    break
                idx = random.randint(0, len(island)-1)
                new_solution = copy.deepcopy(self.solution_template)
                for var_name, variable in new_solution.variables.items():
                    domain_min = variable.domain_min
                    domain_max = variable.domain_max
                    variable.set_value(random.uniform(domain_min, domain_max))
                island[idx] = self.local_search(new_solution)
            self.island_fitness[island_idx] = [sol.calculate_heuristic() or float('inf') for sol in island]
            
            # Check again after reinit
            if len(island) < 2:
                break
            diversity = self.compute_island_diversity(island)
            attempts += 1

    def promote_best_to_next_island(self, island_idx, fitness_threshold):
        if island_idx >= self.num_islands - 1:
            return
        island = self.islands[island_idx]
        fit = self.island_fitness[island_idx]
        
        # Check if island is empty
        if len(island) == 0:
            return
        
        next_island = self.islands[island_idx+1]
        next_fit = self.island_fitness[island_idx+1]
        arr = np.array([f for f in fit if f != float('inf')])
        if len(arr) == 0 or arr.min() > fitness_threshold:
            return
        num_promote = max(1, int(0.1 * len(island)))
        sorted_idx = np.argsort(fit)
        for i in range(num_promote):
            promoted = copy.deepcopy(island[sorted_idx[i]])
            next_island.append(promoted)
            next_fit.append(promoted.calculate_heuristic() or float('inf'))
        for i in sorted(sorted_idx[:num_promote], reverse=True):
            if i < len(island):
                del island[i]
                del fit[i]

    def ga_step(self):
        best_solutions = []
        base_threshold = 0.01
        for island_idx in range(self.num_islands):
            best_solution = self.island_ga_step(island_idx)
            best_solutions.append(best_solution)
            threshold = base_threshold * (1 + 0.5 * island_idx)
            self.force_island_diversity(island_idx, threshold)
            fitness_threshold = 1e-2 * (1 + 0.5 * island_idx)
            self.promote_best_to_next_island(island_idx, fitness_threshold)
        self.update_global_fitness_history()
        
        self.generation_count += 1
        if self.generation_count % self.migration_interval == 0:
            self.migrate_between_islands()
        if self.generation_count % self.visualization_interval == 0:
            self.push_visualization_data()
        return min(best_solutions, key=lambda s: s.calculate_heuristic() or float('inf'))


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
        for var_name, variable in solution.variables.items():
            print(f"{var_name} = {variable.current_value:.6f}")
        print(f"Error: {solution.calculate_heuristic()}")
        return True

    def graduate_solution(self, solution, fitness):
        if not self.is_new_solution(solution):
            return 0  
            
        self.graduated_solutions.append(copy.deepcopy(solution))
        
        total_removed = 0
        for island_idx in range(self.num_islands):
            island = self.islands[island_idx]
            island_fit = self.island_fitness[island_idx]
            
            to_remove = []
            for i, pop_solution in enumerate(island):
                if self.solution_distance(solution, pop_solution) < self.solution_distance_threshold:
                    to_remove.append(i)
            
            for idx in sorted(to_remove, reverse=True):
                if idx < len(island):
                    del island[idx]
                    del island_fit[idx]
            
            total_removed += len(to_remove)
            
            # Replenish if island becomes too small
            min_island_size = max(5, self.island_size // 4)
            if len(island) < min_island_size:
                num_to_add = min_island_size - len(island)
                self.add_random_individuals(island_idx, num_to_add)
        
        return total_removed

    def solve(self):
        stable_count = 0
        
        self.push_visualization_data()
        
        total_population_size = sum(len(island) for island in self.islands)
        
        while time.time() <= self.end_time and total_population_size > self.population_size // 2:
            solution = self.ga_step()
            fitness = solution.calculate_heuristic()
            
            if fitness is None:
                continue
                
            if fitness < self.eps:
                removed_count = self.graduate_solution(solution, fitness)
                if removed_count > 0:
                    stable_count = 0
                    total_population_size = sum(len(island) for island in self.islands)
                    continue            
            
            stable_count += 1
            
            if stable_count >= 20:
                for mutation in self.island_mutations:
                    mutation.rate = min(0.5, mutation.rate * 1.5)  
            if stable_count >= 50:
                for mutation in self.island_mutations:
                    mutation.rate = min(0.8, mutation.rate * 1.2) 
            if stable_count >= 100:
                for island_idx in range(self.num_islands):
                    current_size = len(self.islands[island_idx])
                    if current_size > 0:
                        self.add_random_individuals(island_idx, max(1, current_size // 2))
                
                for mutation in self.island_mutations:
                    mutation.rate = 0.2  
                stable_count = 0
                
            if stable_count % 10 == 0 and stable_count > 0:
                for island_idx in range(self.num_islands):
                    current_size = len(self.islands[island_idx])
                    if current_size > 0:
                        self.add_random_individuals(island_idx, max(1, current_size // 20))
            
            total_population_size = sum(len(island) for island in self.islands)
        
        self.push_visualization_data()
        
        if not self.graduated_solutions and any(self.islands):
            all_solutions_with_fitness = []
            for island_idx in range(self.num_islands):
                for sol, fit in zip(self.islands[island_idx], self.island_fitness[island_idx]):
                    all_solutions_with_fitness.append((sol, fit))
            
            if all_solutions_with_fitness:
                best_solution = min(all_solutions_with_fitness, key=lambda x: x[1])[0]
                self.graduated_solutions.append(best_solution)
                
        return self.graduated_solutions

    def set_mutation_config(self, config):
        for mutation in self.island_mutations:
            mutation.__init__(config)
        
    def set_crossover_config(self, config):
        for crossover in self.island_crossovers:
            crossover.__init__(config)