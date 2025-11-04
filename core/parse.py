from parser.string_to_tree import parse_formula, NodeType, FormulaTree

class Variable:
    def __init__(self, name, excluded_values=None, current_value=None):
        self.name = name
        self.domain_min = float('-inf')
        self.domain_max = float('inf')
        self.excluded_values = excluded_values if excluded_values else []
        self.current_value = current_value
    
    def is_in_domain(self, value):
        if value in self.excluded_values:
            return False
        
        return value >= self.domain_min and value <= self.domain_max
    
    def set_value(self, value):
        if self.is_in_domain(value):
            self.current_value = value
            return True
        return False
    
    def set_domain(self, domain_min=None, domain_max=None):
        if domain_min is not None:
            self.domain_min = domain_min
        if domain_max is not None:
            self.domain_max = domain_max
    
    def __str__(self):
        domain_str = f"[{'-∞' if self.domain_min == float('-inf') else self.domain_min}, " \
                     f"{'+∞' if self.domain_max == float('inf') else self.domain_max}]"
        if self.excluded_values:
            domain_str += f" excluding {self.excluded_values}"
        return f"Variable({self.name}, domain={domain_str}, value={self.current_value})"


class Solution:
    def __init__(self, variables=None, left_tree=None, right_tree=None, left_expr=None, right_expr=None):
        self.variables = variables if variables else {}
        self.left_tree = left_tree
        self.right_tree = right_tree
        self.left_expr = left_expr
        self.right_expr = right_expr
    
    def add_variable(self, variable):
        self.variables[variable.name] = variable
    
    def evaluate_expression(self, tree):
        if not tree:
            return 0
            
        var_values = {name: var.current_value for name, var in self.variables.items()
                     if var.current_value is not None}
        
        if len(var_values) != len(self.variables):
            return None
            
        try:
            return tree.evaluate(var_values)
        except Exception as e:
            print(f"Error evaluating expression: {e}")
            return None
    
    def calculate_heuristic(self):
        lhs = self.evaluate_expression(self.left_tree)
        rhs = self.evaluate_expression(self.right_tree)
        
        if lhs is None or rhs is None:
            return None
            
        return abs(lhs - rhs)
    
    def __str__(self):
        result = f"Solution with {len(self.variables)} variables:\n"
        for var in self.variables.values():
            result += f"  {var}\n"
        result += f"Heuristic: abs({self.left_expr} - {self.right_expr})"
        return result

def create_solution_from_trees(left_tree, right_tree=None):

    variables = {}
    
    extract_variables(left_tree.root, variables)
    if right_tree:
        extract_variables(right_tree.root, variables)
    
    var_objects = {
        name: Variable(name) for name in variables
    }
    
    left_expr = str(left_tree)
    right_expr = str(right_tree) if right_tree else "0"
    
    if right_tree is None:
        from parser.string_to_tree import NumberNode
        right_tree = FormulaTree(NumberNode(0))
    
    return Solution(var_objects, left_tree, right_tree, left_expr, right_expr)

def extract_variables(node, variables_set):
    if node.node_type == NodeType.VARIABLE:
        variables_set[node.name] = True
    elif node.node_type == NodeType.BINARY_OP:
        extract_variables(node.left, variables_set)
        extract_variables(node.right, variables_set)
    elif node.node_type == NodeType.FUNCTION:
        extract_variables(node.argument, variables_set)


if __name__ == "__main__":
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
    
    print("\nEnter values for variables (leave blank to skip):")
    for var_name in solution.variables:
        value_str = input(f"{var_name} = ")
        if value_str.strip():
            try:
                value = float(value_str)
                solution.variables[var_name].set_value(value)
            except ValueError:
                print(f"Invalid number: {value_str}")
    
    heuristic = solution.calculate_heuristic()
    if heuristic is not None:
        print(f"\nHeuristic value: {heuristic}")
    else:
        print("\nCannot calculate heuristic (missing variable values)")