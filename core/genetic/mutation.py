import random
import math
from copy import deepcopy
from parse import Solution, Variable

class Mutation:
    
    def __init__(self, config=None):
        self.config = config or {}
        self.rate = self.config.get('rate', 0.1)  
        self.method = self.config.get('method', 'gaussian')
        self.scale = self.config.get('scale', 0.1) 
        self.uniform_range = self.config.get('uniform_range', 0.1)  
    
    def forward(self, solution):
        mutated = deepcopy(solution)
        
        for var_name, variable in mutated.variables.items():
            if random.random() > self.rate:
                continue
                
            if variable.current_value is None:
                continue
            
            domain_min = variable.domain_min
            domain_max = variable.domain_max
            
            if domain_min == float('-inf'):
                domain_min = variable.current_value - abs(variable.current_value) - 10
            if domain_max == float('inf'):
                domain_max = variable.current_value + abs(variable.current_value) + 10
            
            domain_size = domain_max - domain_min
            
            if self.method == 'gaussian':
                self._gaussian_mutation(variable, domain_size)
            elif self.method == 'uniform':
                self._uniform_mutation(variable, domain_min, domain_max)
            else: 
                self._random_reset_mutation(variable, domain_min, domain_max)
        
        return mutated
    
    def _gaussian_mutation(self, variable, domain_size):
        """Apply Gaussian mutation to a variable"""
        current_value = variable.current_value
        std_dev = domain_size * self.scale
        
        delta = random.gauss(0, std_dev)
        new_value = current_value + delta
        
        attempts = 0
        while not variable.is_in_domain(new_value) and attempts < 5:
            delta = random.gauss(0, std_dev)
            new_value = current_value + delta
            attempts += 1
        
        if variable.is_in_domain(new_value):
            variable.current_value = new_value
    
    def _uniform_mutation(self, variable, domain_min, domain_max):
        current_value = variable.current_value
        domain_size = domain_max - domain_min
        
        max_delta = domain_size * self.uniform_range
        delta = random.uniform(-max_delta, max_delta)
        new_value = current_value + delta
        
        attempts = 0
        while not variable.is_in_domain(new_value) and attempts < 5:
            delta = random.uniform(-max_delta, max_delta)
            new_value = current_value + delta
            attempts += 1
        
        if variable.is_in_domain(new_value):
            variable.current_value = new_value
    
    def _random_reset_mutation(self, variable, domain_min, domain_max):
        excluded = set(variable.excluded_values)
        
        for _ in range(10): 
            new_value = random.uniform(domain_min, domain_max)
            if variable.is_in_domain(new_value):
                variable.current_value = new_value
                break