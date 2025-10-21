import random
import math
from copy import deepcopy
from parse import Solution, Variable

class Crossover:    
    def __init__(self, config=None):
        self.config = config or {}
        self.method = self.config.get('method', 'arithmetic')
        self.rate = self.config.get('rate', 0.8)  
        self.alpha = self.config.get('alpha', 0.5) 
    
    def forward(self, parent1, parent2):
        if random.random() > self.rate:
            return parent1, parent2
        
        child1 = deepcopy(parent1)
        child2 = deepcopy(parent2)
        
        var_names = set(parent1.variables.keys()) | set(parent2.variables.keys())
        
        if self.method == 'single_point':
            self._single_point_crossover(child1, child2, var_names)
        elif self.method == 'uniform':
            self._uniform_crossover(child1, child2, var_names)
        else: 
            self._arithmetic_crossover(child1, child2, var_names)
        
        return child1, child2
    
    def _single_point_crossover(self, child1, child2, var_names):
        var_list = list(var_names)
        if not var_list:
            return
            
        crossover_point = random.randint(0, len(var_list) - 1)
        
        for i in range(crossover_point, len(var_list)):
            var_name = var_list[i]
            if var_name in child1.variables and var_name in child2.variables:
                val1 = child1.variables[var_name].current_value
                val2 = child2.variables[var_name].current_value
                
                if val1 is not None and child2.variables[var_name].is_in_domain(val1):
                    child2.variables[var_name].current_value = val1
                    
                if val2 is not None and child1.variables[var_name].is_in_domain(val2):
                    child1.variables[var_name].current_value = val2
    
    def _uniform_crossover(self, child1, child2, var_names):
        for var_name in var_names:
            if var_name in child1.variables and var_name in child2.variables:
                if random.random() < 0.5:
                    val1 = child1.variables[var_name].current_value
                    val2 = child2.variables[var_name].current_value
                    
                    if val1 is not None and child2.variables[var_name].is_in_domain(val1):
                        child2.variables[var_name].current_value = val1
                        
                    if val2 is not None and child1.variables[var_name].is_in_domain(val2):
                        child1.variables[var_name].current_value = val2
    
    def _arithmetic_crossover(self, child1, child2, var_names):
        for var_name in var_names:
            if var_name in child1.variables and var_name in child2.variables:
                val1 = child1.variables[var_name].current_value
                val2 = child2.variables[var_name].current_value
                
                if val1 is None or val2 is None:
                    continue
                
                new_val1 = self.alpha * val1 + (1 - self.alpha) * val2
                new_val2 = self.alpha * val2 + (1 - self.alpha) * val1
                
                if child1.variables[var_name].is_in_domain(new_val1):
                    child1.variables[var_name].current_value = new_val1
                
                if child2.variables[var_name].is_in_domain(new_val2):
                    child2.variables[var_name].current_value = new_val2