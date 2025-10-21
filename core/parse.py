from parser.string_to_tree import parse_formula, NodeType, FormulaTree

class Variable:
    """
    Represents a variable in a solution with domain constraints and current value.
    """
    def __init__(self, name, excluded_values=None, current_value=None):
        """
        Initialize a variable with infinite domain by default and optional current value.
        
        Args:
            name (str): Name of the variable
            excluded_values (list): Values to exclude from domain
            current_value: Current assigned value of the variable
        """
        self.name = name
        self.domain_min = float('-inf')  # Negative infinity by default
        self.domain_max = float('inf')   # Positive infinity by default
        self.excluded_values = excluded_values if excluded_values else []
        self.current_value = current_value
    
    def is_in_domain(self, value):
        """Check if a value is within the domain constraints"""
        if value in self.excluded_values:
            return False
        
        return value >= self.domain_min and value <= self.domain_max
    
    def set_value(self, value):
        """Set variable value if it's in domain"""
        if self.is_in_domain(value):
            self.current_value = value
            return True
        return False
    
    def set_domain(self, domain_min=None, domain_max=None):
        """Update the domain range of the variable"""
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
    """
    Represents a solution set consisting of variables and a heuristic function.
    """
    def __init__(self, variables=None, left_tree=None, right_tree=None, left_expr=None, right_expr=None):
        """
        Initialize a solution with variables and formula trees for heuristic.
        
        Args:
            variables (dict): Dictionary of variable name to Variable objects
            left_tree (FormulaTree): Left-hand side formula tree
            right_tree (FormulaTree): Right-hand side formula tree
            left_expr (str): String representation of left expression (for display)
            right_expr (str): String representation of right expression (for display)
        """
        self.variables = variables if variables else {}
        self.left_tree = left_tree
        self.right_tree = right_tree
        self.left_expr = left_expr
        self.right_expr = right_expr
    
    def add_variable(self, variable):
        """Add a variable to the solution"""
        self.variables[variable.name] = variable
    
    def evaluate_expression(self, tree):
        """Evaluate expression using FormulaTree with current variable values"""
        if not tree:
            return 0
            
        # Create a dictionary of variable values for evaluation
        var_values = {name: var.current_value for name, var in self.variables.items()
                     if var.current_value is not None}
        
        # Only evaluate if all variables have values
        if len(var_values) != len(self.variables):
            return None
            
        try:
            return tree.evaluate(var_values)
        except Exception as e:
            print(f"Error evaluating expression: {e}")
            return None
    
    def calculate_heuristic(self):
        """Calculate heuristic as abs(LHS - RHS)"""
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
    """
    Create a Solution object from FormulaTree objects.
    
    Args:
        left_tree (FormulaTree): The left side of the equation
        right_tree (FormulaTree, optional): The right side of the equation.
            If None, right side is considered as 0.
            
    Returns:
        Solution: A Solution object representing the equation
    """
    # Initialize variables dictionary
    variables = {}
    
    # Extract variables from both trees
    extract_variables(left_tree.root, variables)
    if right_tree:
        extract_variables(right_tree.root, variables)
    
    # Create Variable objects
    var_objects = {
        name: Variable(name) for name in variables
    }
    
    # Create string representations for expressions
    left_expr = str(left_tree)
    right_expr = str(right_tree) if right_tree else "0"
    
    # Create a dummy tree for right side if needed
    if right_tree is None:
        from parser.string_to_tree import NumberNode
        right_tree = FormulaTree(NumberNode(0))
    
    # Create and return Solution object with both strings and trees
    return Solution(var_objects, left_tree, right_tree, left_expr, right_expr)

def extract_variables(node, variables_set):
    """
    Recursively extract variable names from a formula tree node.
    
    Args:
        node: The current node in the formula tree
        variables_set: Dictionary to store found variable names
    """
    if node.node_type == NodeType.VARIABLE:
        variables_set[node.name] = True
    elif node.node_type == NodeType.BINARY_OP:
        extract_variables(node.left, variables_set)
        extract_variables(node.right, variables_set)
    elif node.node_type == NodeType.FUNCTION:
        extract_variables(node.argument, variables_set)
    # NUMBER nodes don't contain variables, so we skip them


# Modify the main block

if __name__ == "__main__":
    input_str = input("Enter equation: ")
    input_str = input_str.replace(" ", "")
    assert "=" in input_str, "Equation must contain '=' sign"

    left, right = input_str.split("=")
    formula_tree_left = parse_formula(left)
    formula_tree_right = parse_formula(right)
    
    # Print the formula trees
    print("\nLeft side formula tree:")
    formula_tree_left.print_tree()
    print("\nRight side formula tree:")
    formula_tree_right.print_tree()
    
    # Create solution from trees
    solution = create_solution_from_trees(formula_tree_left, formula_tree_right)
    print("\nCreated solution:")
    print(solution)
    
    # Optionally, get variable values from user
    print("\nEnter values for variables (leave blank to skip):")
    for var_name in solution.variables:
        value_str = input(f"{var_name} = ")
        if value_str.strip():
            try:
                value = float(value_str)
                solution.variables[var_name].set_value(value)
            except ValueError:
                print(f"Invalid number: {value_str}")
    
    # Calculate and display heuristic
    heuristic = solution.calculate_heuristic()
    if heuristic is not None:
        print(f"\nHeuristic value: {heuristic}")
    else:
        print("\nCannot calculate heuristic (missing variable values)")