import numpy as np
import copy
from typing import List, Dict, Tuple, Callable
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.parse import Solution, Variable, create_solution_from_trees
from core.parser.string_to_tree import parse_formula


class BenchmarkProblem:
    """
    Represents a single benchmark problem with metadata.
    """
    def __init__(self, name: str, equation: str, category: str, 
                 dimension: int, known_solutions: List[Dict[str, float]] = None,
                 difficulty: str = "medium"):
        self.name = name
        self.equation = equation
        self.category = category  # "classical" or "scalable"
        self.dimension = dimension
        self.known_solutions = known_solutions or []
        self.difficulty = difficulty
        
    def to_solution(self) -> Solution:
        """Convert benchmark problem to Solution object"""
        if '=' not in self.equation:
            raise ValueError(f"Equation must contain '=': {self.equation}")
        
        left_str, right_str = [s.strip() for s in self.equation.split('=', 1)]
        left_tree = parse_formula(left_str)
        right_tree = parse_formula(right_str or "0")
        
        solution = create_solution_from_trees(left_tree, right_tree)
        
        # Auto-detect and set domains using the logic from app.py
        for var_name in solution.variables:
            domain_min, domain_max = self._detect_domain_for_var(
                left_tree, right_tree, var_name
            )
            solution.variables[var_name].set_domain(domain_min, domain_max)
        
        return solution
    
    def _detect_domain_for_var(self, left_tree, right_tree, var_name):
        """Domain detection logic from app.py"""
        from core.parser.string_to_tree import (
            FunctionNode, BinaryOpNode, NumberNode, VariableNode
        )
        
        lower = float('-inf')
        upper = float('inf')

        def walk(node, target_var):
            nonlocal lower, upper
            if isinstance(node, FunctionNode):
                arg = node.argument
                if node.function_name == 'log':
                    if isinstance(arg, BinaryOpNode) and arg.operator == '-':
                        if isinstance(arg.left, NumberNode) and isinstance(arg.right, VariableNode):
                            if arg.right.name == target_var:
                                upper = min(upper, arg.left.value - 1e-9)
                        elif isinstance(arg.left, VariableNode) and isinstance(arg.right, NumberNode):
                            if arg.left.name == target_var:
                                lower = max(lower, arg.right.value + 1e-9)
                    elif isinstance(arg, VariableNode) and arg.name == target_var:
                        lower = max(lower, 1e-9)
                    elif isinstance(arg, BinaryOpNode) and arg.operator == '+':
                        if isinstance(arg.left, VariableNode) and isinstance(arg.right, NumberNode):
                            if arg.left.name == target_var:
                                lower = max(lower, -arg.right.value + 1e-9)
                
                elif node.function_name == 'sqrt':
                    if isinstance(arg, BinaryOpNode) and arg.operator == '+':
                        if isinstance(arg.left, VariableNode) and isinstance(arg.right, NumberNode):
                            if arg.left.name == target_var:
                                lower = max(lower, -arg.right.value + 1e-9)
                    elif isinstance(arg, BinaryOpNode) and arg.operator == '-':
                        if isinstance(arg.left, NumberNode) and isinstance(arg.right, VariableNode):
                            if arg.right.name == target_var:
                                upper = min(upper, arg.left.value - 1e-9)
                        elif isinstance(arg.left, VariableNode) and isinstance(arg.right, NumberNode):
                            if arg.left.name == target_var:
                                lower = max(lower, arg.right.value + 1e-9)
                    elif isinstance(arg, VariableNode) and arg.name == target_var:
                        lower = max(lower, 1e-9)
                        
                elif node.function_name in ('arcsin', 'arccos'):
                    if isinstance(arg, VariableNode) and arg.name == target_var:
                        lower = max(lower, -1)
                        upper = min(upper, 1)
                        
            elif hasattr(node, 'left') and hasattr(node, 'right'):
                walk(node.left, target_var)
                walk(node.right, target_var)
            elif hasattr(node, 'argument'):
                walk(node.argument, target_var)

        walk(left_tree.root, var_name)
        walk(right_tree.root, var_name)

        if lower == float('-inf'):
            lower = -100
        if upper == float('inf'):
            upper = 100

        if lower >= upper:
            lower = -100
            upper = 100

        return lower, upper


class BenchmarkGenerator:
    """
    Generates benchmark test suites for equation solving.
    """
    
    def __init__(self):
        self.benchmarks = []
        
    def generate_classical_suite(self) -> List[BenchmarkProblem]:
        """
        Tier A: Classical nonlinear systems from numerical analysis literature.
        These are well-known problems used to validate root-finding algorithms.
        """
        classical = [
            # Simple polynomial equations
            BenchmarkProblem(
                name="Quadratic",
                equation="x^2-4=0",
                category="classical",
                dimension=1,
                known_solutions=[{"x": 2.0}, {"x": -2.0}],
                difficulty="easy"
            ),
            
            BenchmarkProblem(
                name="Cubic",
                equation="x^3-6*x^2+11*x-6=0",
                category="classical",
                dimension=1,
                known_solutions=[{"x": 1.0}, {"x": 2.0}, {"x": 3.0}],
                difficulty="easy"
            ),
            
            # Transcendental equations
            BenchmarkProblem(
                name="Exponential_Simple",
                equation="2^x-8=0",
                category="classical",
                dimension=1,
                known_solutions=[{"x": 3.0}],
                difficulty="easy"
            ),
            
            BenchmarkProblem(
                name="Trigonometric_Basic",
                equation="sin(x)=0.5",
                category="classical",
                dimension=1,
                known_solutions=[{"x": 0.5236}],  # π/6
                difficulty="medium"
            ),
            
            # The original example from the paper
            BenchmarkProblem(
                name="Mixed_Transcendental",
                equation="sqrt(x+25)+log(49-x)=25",
                category="classical",
                dimension=1,
                known_solutions=[{"x": 24.0}],
                difficulty="hard"
            ),
            
            # Multi-variable systems
            BenchmarkProblem(
                name="Rosenbrock_Roots",
                equation="10*(y-x^2)+(1-x)=0",
                category="classical",
                dimension=2,
                known_solutions=[{"x": 1.0, "y": 1.0}],
                difficulty="hard"
            ),
            
            BenchmarkProblem(
                name="Circle_Line_Intersection",
                equation="x^2+y^2-25=0",
                category="classical",
                dimension=2,
                known_solutions=[{"x": 3.0, "y": 4.0}, {"x": 4.0, "y": 3.0}],
                difficulty="medium"
            ),
            
            # Powell's singular function (famous benchmark)
            BenchmarkProblem(
                name="Powell_Singular_2D",
                equation="x+10*y=0",
                category="classical",
                dimension=2,
                known_solutions=[{"x": 0.0, "y": 0.0}],
                difficulty="hard"
            ),
        ]
        
        self.benchmarks.extend(classical)
        return classical
    
    def generate_scalable_suite(self) -> List[BenchmarkProblem]:
        """
        Tier B: Scalable problems that test dimensional scalability.
        These problems can be extended to N variables.
        """
        scalable = []
        
        # Broyden Tridiagonal System (2D, 3D, 5D versions)
        for n in [2, 3, 5]:
            # Generate equation: sum of coupled terms
            # Example for n=3: x*(3-2*x) - y - 2*z + 1 + y*(3-2*y) - x - 2*0 + 1 + z*(3-2*z) - y - 2*0 + 1 = 0
            var_names = ['x', 'y', 'z', 'w', 'v'][:n]
            
            terms = []
            for i in range(n):
                curr_var = var_names[i]
                prev_var = var_names[i-1] if i > 0 else "0"
                next_var = var_names[i+1] if i < n-1 else "0"
                
                term = f"{curr_var}*(3-2*{curr_var})-{prev_var}-2*{next_var}+1"
                terms.append(f"({term})")
            
            equation = "+".join(terms) + "=0"
            
            scalable.append(BenchmarkProblem(
                name=f"Broyden_Tridiagonal_{n}D",
                equation=equation,
                category="scalable",
                dimension=n,
                known_solutions=[],  # Complex analytical solution
                difficulty="hard"
            ))
        
        # Sum of squares system (tests curse of dimensionality)
        for n in [2, 3, 4]:
            var_names = ['x', 'y', 'z', 'w'][:n]
            terms = [f"{v}^2" for v in var_names]
            equation = "+".join(terms) + f"-{n}=0"
            
            # Known solution: all variables = 1 or -1
            scalable.append(BenchmarkProblem(
                name=f"Sum_of_Squares_{n}D",
                equation=equation,
                category="scalable",
                dimension=n,
                known_solutions=[{v: 1.0 for v in var_names}],
                difficulty="medium"
            ))
        
        self.benchmarks.extend(scalable)
        return scalable
    
    def generate_full_suite(self) -> List[BenchmarkProblem]:
        """Generate complete benchmark suite"""
        self.benchmarks = []
        self.generate_classical_suite()
        self.generate_scalable_suite()
        return self.benchmarks
    
    def get_benchmark_by_name(self, name: str) -> BenchmarkProblem:
        """Retrieve specific benchmark by name"""
        for bench in self.benchmarks:
            if bench.name == name:
                return bench
        raise ValueError(f"Benchmark '{name}' not found")
    
    def get_benchmarks_by_category(self, category: str) -> List[BenchmarkProblem]:
        """Get all benchmarks in a category"""
        return [b for b in self.benchmarks if b.category == category]


if __name__ == "__main__":
    # Test the benchmark generator
    generator = BenchmarkGenerator()
    suite = generator.generate_full_suite()
    
    print(f"Generated {len(suite)} benchmark problems:\n")
    
    for bench in suite:
        print(f"Name: {bench.name}")
        print(f"Category: {bench.category}")
        print(f"Dimension: {bench.dimension}")
        print(f"Equation: {bench.equation}")
        print(f"Difficulty: {bench.difficulty}")
        print(f"Known solutions: {len(bench.known_solutions)}")
        
        # Test conversion to Solution object
        try:
            solution = bench.to_solution()
            print(f"✓ Successfully converted to Solution object")
            print(f"  Variables: {list(solution.variables.keys())}")
            for var_name, var in solution.variables.items():
                print(f"  {var_name}: domain [{var.domain_min:.2f}, {var.domain_max:.2f}]")
        except Exception as e:
            print(f"✗ Error converting to Solution: {e}")
        
        print("-" * 60)