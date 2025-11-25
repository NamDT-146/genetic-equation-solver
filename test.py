import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.parse import parse_formula, create_solution_from_trees
from core.ga_solver import GASolver

def input_equation():
    input_str = input("Enter equation: ")
    input_str = input_str.replace(" ", "")
    assert "=" in input_str, "Equation must contain '=' sign"

    left, right = input_str.split("=")
    formula_tree_left = parse_formula(left)
    formula_tree_right = parse_formula(right)

    print("\nLeft side formula tree:")
    formula_tree_left.print_tree()
    print("\nRight side formula tree:")
    formula_tree_right.print_tree()

    solution = create_solution_from_trees(formula_tree_left, formula_tree_right)
    print("\nCreated solution:")
    print(solution)

    return (formula_tree_left, formula_tree_right, solution)

if __name__ == "__main__":
    formula_tree_left, formula_tree_right, solution = input_equation()
    solution.variables['x'].set_domain(-100, 100)

    solver = GASolver(
        solution_template=solution,
        population_size=200,
        generations=1000,
        tournament_size=5,
        elitism_size=5,
        time_limit=30
    )

    best_solutions = solver.solve()

    print(f"Equation: {formula_tree_left} = {formula_tree_right}")
    print(f"Best solutions found:")
    for sol in best_solutions:
        for var_name, variable in sol.variables.items():
            print(f"{var_name} = {variable.current_value:.6f}")
        print(f"Error: {sol.calculate_heuristic()}")
        print("-----")