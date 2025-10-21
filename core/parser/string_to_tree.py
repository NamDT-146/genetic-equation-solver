from enum import Enum
import math

class NodeType(Enum):
    NUMBER = "NUMBER"
    VARIABLE = "VARIABLE"
    BINARY_OP = "BINARY_OP"
    FUNCTION = "FUNCTION"

class Node:
    """Base class for all nodes in the formula tree"""
    def __init__(self, node_type: NodeType):
        self.node_type = node_type
    
    def evaluate(self, variable_values=None):
        """Evaluate the node with provided variable values"""
        raise NotImplementedError("Each node must implement evaluate method")
    
    def __str__(self):
        raise NotImplementedError("Each node must implement __str__ method")

class NumberNode(Node):
    """Leaf node representing a numeric value"""
    def __init__(self, value: float):
        super().__init__(NodeType.NUMBER)
        self.value = value
    
    def evaluate(self, variable_values=None):
        return self.value
    
    def __str__(self):
        return str(self.value)

class VariableNode(Node):
    """Leaf node representing a variable"""
    def __init__(self, name: str):
        super().__init__(NodeType.VARIABLE)
        self.name = name
    
    def evaluate(self, variable_values=None):
        if variable_values is None or self.name not in variable_values:
            raise ValueError(f"No value provided for variable '{self.name}'")
        return variable_values[self.name]
    
    def __str__(self):
        return self.name

class BinaryOpNode(Node):
    """Node representing a binary operation (+, -, *, /, ^)"""
    def __init__(self, operator: str, left: Node, right: Node):
        super().__init__(NodeType.BINARY_OP)
        self.operator = operator
        self.left = left
        self.right = right
        
        # Define operation functions
        self.op_funcs = {
            '+': lambda x, y: x + y,
            '-': lambda x, y: x - y,
            '*': lambda x, y: x * y,
            '/': lambda x, y: x / y,
            '^': lambda x, y: pow(x, y)
        }
    
    def evaluate(self, variable_values=None):
        if self.operator not in self.op_funcs:
            raise ValueError(f"Unknown operator: {self.operator}")
        
        left_val = self.left.evaluate(variable_values)
        right_val = self.right.evaluate(variable_values)
        return self.op_funcs[self.operator](left_val, right_val)
    
    def __str__(self):
        return f"({self.left} {self.operator} {self.right})"

class FunctionNode(Node):
    """Node representing a function call (sin, cos, etc.)"""
    def __init__(self, function_name: str, argument: Node):
        super().__init__(NodeType.FUNCTION)
        self.function_name = function_name
        self.argument = argument
        
        # Define function mappings
        self.func_map = {
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'arcsin': math.asin,
            'arccos': math.acos, 
            'arctan': math.atan,
            'sqrt': math.sqrt,
            'log': math.log
        }
    
    def evaluate(self, variable_values=None):
        if self.function_name not in self.func_map:
            raise ValueError(f"Unknown function: {self.function_name}")
        
        arg_val = self.argument.evaluate(variable_values)
        return self.func_map[self.function_name](arg_val)
    
    def __str__(self):
        return f"{self.function_name}({self.argument})"

class FormulaTree:
    def __init__(self, root: Node):
        self.root = root
    
    def evaluate(self, variable_values=None):
        return self.root.evaluate(variable_values)
    
    def __str__(self):
        return str(self.root)
    
    def _build_graph(self, node, graph, node_labels, parent_id=None, node_id=None):
        if node_id is None:
            node_id = id(node)
        
        if node.node_type == NodeType.NUMBER:
            node_labels[node_id] = str(node.value)
            graph.add_node(node_id)
        elif node.node_type == NodeType.VARIABLE:
            node_labels[node_id] = node.name
            graph.add_node(node_id)
        elif node.node_type == NodeType.BINARY_OP:
            node_labels[node_id] = node.operator
            graph.add_node(node_id)
            
            left_id = id(node.left)
            right_id = id(node.right)
            
            self._build_graph(node.left, graph, node_labels, node_id, left_id)
            self._build_graph(node.right, graph, node_labels, node_id, right_id)
        elif node.node_type == NodeType.FUNCTION:
            node_labels[node_id] = node.function_name
            graph.add_node(node_id)
            
            arg_id = id(node.argument)
            self._build_graph(node.argument, graph, node_labels, node_id, arg_id)
        
        if parent_id is not None:
            graph.add_edge(parent_id, node_id)


    def print_tree(self):
        """Print a text-based visualization of the formula tree to the terminal"""
        print("Formula Tree:")
        self._print_node(self.root, "", True)

    def _print_node(self, node, prefix, is_last):
        """Recursively print a node and its children with proper formatting"""
        # Create the line prefix with appropriate branching characters
        line_prefix = prefix + ("└── " if is_last else "├── ")
        
        # Print the current node
        if node.node_type == NodeType.NUMBER:
            print(f"{line_prefix}Number: {node.value}")
        elif node.node_type == NodeType.VARIABLE:
            print(f"{line_prefix}Variable: {node.name}")
        elif node.node_type == NodeType.BINARY_OP:
            print(f"{line_prefix}Operator: {node.operator}")
            
            # Prepare the next prefix for children
            next_prefix = prefix + ("    " if is_last else "│   ")
            
            # Print children
            self._print_node(node.left, next_prefix, False)
            self._print_node(node.right, next_prefix, True)
        elif node.node_type == NodeType.FUNCTION:
            print(f"{line_prefix}Function: {node.function_name}")
            
            # Prepare the next prefix for children
            next_prefix = prefix + ("    " if is_last else "│   ")
            
            # Print the argument
            self._print_node(node.argument, next_prefix, True)

class FormulaParser:
    def __init__(self):
        self.precedence = {
            '+': 1,
            '-': 1,
            '*': 2,
            '/': 2,
            '^': 3
        }
        
        self.functions = {
            'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'sqrt', 'log'
        }
    
    def parse(self, expression: str) -> FormulaTree:
        expression = expression.replace(' ', '').lower()
        
        node = self._parse_expression(expression)
        
        return FormulaTree(node)
    
    def _parse_expression(self, expr: str) -> Node:
        min_precedence = float('inf')
        op_index = -1
        paren_level = 0
        
        for i, char in enumerate(expr):
            if char == '(':
                paren_level += 1
            elif char == ')':
                paren_level -= 1
            elif paren_level == 0 and char in self.precedence:
                if char == '-' and (i == 0 or expr[i-1] in self.precedence or expr[i-1] == '('):
                    continue 
                
                if self.precedence[char] <= min_precedence:
                    min_precedence = self.precedence[char]
                    op_index = i
        
        if op_index != -1:
            left_expr = expr[:op_index]
            operator = expr[op_index]
            right_expr = expr[op_index+1:]
            
            left_node = self._parse_expression(left_expr)
            right_node = self._parse_expression(right_expr)
            
            return BinaryOpNode(operator, left_node, right_node)
        
        for func in self.functions:
            if expr.startswith(func + '('):
                paren_level = 0
                closing_index = -1
                
                for i, char in enumerate(expr[len(func)+1:], len(func)+1):
                    if char == '(':
                        paren_level += 1
                    elif char == ')':
                        if paren_level == 0:
                            closing_index = i
                            break
                        paren_level -= 1
                
                if closing_index == -1:
                    raise ValueError(f"Missing closing parenthesis in function call: {expr}")
                
                arg_expr = expr[len(func)+1:closing_index]
                
                if closing_index != len(expr)-1:
                    raise ValueError(f"Unexpected content after function call: {expr}")
                
                arg_node = self._parse_expression(arg_expr)
                
                return FunctionNode(func, arg_node)
        
        if expr.startswith('(') and expr.endswith(')'):
            paren_level = 0
            for i, char in enumerate(expr):
                if char == '(':
                    paren_level += 1
                elif char == ')':
                    paren_level -= 1
                    if paren_level == 0 and i < len(expr) - 1:
                        break
            
            if paren_level == 0:
                return self._parse_expression(expr[1:-1])
        
        if expr.startswith('-'):
            right_node = self._parse_expression(expr[1:])
            return BinaryOpNode('-', NumberNode(0), right_node)
        
        if len(expr) == 1 and 'a' <= expr <= 'z':
            return VariableNode(expr)
        
        try:
            value = float(expr)
            return NumberNode(value)
        except ValueError:
            pass
        
        raise ValueError(f"Could not parse expression: {expr}")

def parse_formula(formula_str: str) -> FormulaTree:
    parser = FormulaParser()
    return parser.parse(formula_str)

# Replace or modify the example usage in the main block

if __name__ == "__main__":
    # Parse some example formulas
    examples = [
        "x + y",
        "sin(x) * cos(y)",
        "sqrt(x^2 + y^2)",
        "log(x) + 5",
        "(a + b) / (c - d)",
        "sin(arctan(y/x))"
    ]
    
    for example in examples:
        print(f"\n{'='*50}")
        print(f"Parsing: {example}")
        try:
            tree = parse_formula(example)
            print(f"String representation: {tree}")
            
            # Try evaluation with some values
            values = {'x': 2, 'y': 3, 'a': 4, 'b': 5, 'c': 6, 'd': 1}
            try:
                result = tree.evaluate(values)
                print(f"Evaluated with {values}: {result}")
            except Exception as e:
                print(f"Evaluation error: {e}")
            
            # Print tree to terminal instead of graphical visualization
            tree.print_tree()
        except Exception as e:
            print(f"Parsing error: {e}")