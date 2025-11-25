import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import time
import os
import sys

# Add project root to sys.path for icon loading
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load icon
icon_path = os.path.join(project_root, "icon.ico")  # .ico for Windows
if os.path.exists(icon_path):
    pass  # Will be used
else:
    icon_path = os.path.join(project_root, "icon.png")  # Fallback to .png

from core.parse import create_solution_from_trees, parse_formula
from core.ga_solver import GASolver
from core.parser.string_to_tree import NodeType, FunctionNode, BinaryOpNode, NumberNode, VariableNode


class SolverGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GA Equation Solver")

        # --- SET CUSTOM ICON ---
        if os.path.exists(icon_path):
            self.root.iconbitmap(icon_path) if icon_path.endswith(".ico") else self.root.iconphoto(True, tk.PhotoImage(file=icon_path))

        self.root.geometry("1200x800")

        self.solver = None
        self.solver_thread = None
        self.history = []
        self.domain_entries = {}  # Store domain entry widgets
        self.detected_domains = {}  # Store auto-detected domains

        self._build_ui()

    def _build_ui(self):
        main_pane = ttk.PanedWindow(self.root, orient="horizontal")
        main_pane.pack(fill="both", expand=True, padx=10, pady=10)

        left = ttk.Frame(main_pane)
        main_pane.add(left, weight=1)

        eq_f = ttk.LabelFrame(left, text="Equation", padding=10)
        eq_f.pack(fill="x", pady=5)
        self.equation_var = tk.StringVar(value="sqrt(x+25)+log(49-x)=25")
        entry = ttk.Entry(eq_f, textvariable=self.equation_var, font=("Consolas", 13), width=60)
        entry.pack(fill="x", expand=True)
        
        # Add analyze button next to equation
        analyze_btn = ttk.Button(eq_f, text="Analyze Equation", command=self.analyze_equation)
        analyze_btn.pack(fill="x", pady=(5, 0))

        btn_f = ttk.LabelFrame(left, text="Functions", padding=8)
        btn_f.pack(fill="x", pady=5)

        buttons = [
            ('sin(', 'cos(', 'tan(', 'log('),
            ('arcsin(', 'arccos(', 'arctan(', 'sqrt('),
            ('^', '(', ')', 'π'),
            ('x', 'y', 'z', 'DEL'),
            ('7', '8', '9', '/'),
            ('4', '5', '6', '*'),
            ('1', '2', '3', '-'),
            ('0', '.', '=', '+'),
            ('CLR',)
        ]

        for r, row in enumerate(buttons):
            for c, txt in enumerate(row):
                bg = '#90ee90' if txt in ('x', 'y', 'z') else '#ff6b6b' if txt in ('DEL', 'CLR') else '#e0e0e0'
                cmd = lambda t=txt: self._insert(t)
                b = tk.Button(btn_f, text=txt, bg=bg, width=6, height=1, command=cmd)
                b.grid(row=r, column=c, padx=1, pady=1, sticky="nsew")
            for c in range(len(row)):
                btn_f.grid_columnconfigure(c, weight=1)

        # Domain configuration frame
        self.domain_frame = ttk.LabelFrame(left, text="Variable Domains", padding=10)
        self.domain_frame.pack(fill="x", pady=5)
        
        ttk.Label(self.domain_frame, text="Click 'Analyze Equation' to detect variables and set domains", 
                  foreground="gray").pack(pady=10)

        param_f = ttk.LabelFrame(left, text="GA Settings", padding=10)
        param_f.pack(fill="x", pady=5)

        labels = ["Population:", "Time (s):", "Tournament:", "Elitism:"]
        defaults = [400, 60, 5, 5]
        self.param_vars = [tk.IntVar(value=v) for v in defaults]

        for i, (lbl, var) in enumerate(zip(labels, self.param_vars)):
            ttk.Label(param_f, text=lbl).grid(row=i//2, column=(i%2)*2, sticky="w", padx=5)
            spin = ttk.Spinbox(param_f, from_=1, to=9999, textvariable=var, width=10)
            spin.grid(row=i//2, column=(i%2)*2 + 1, padx=5, pady=2)

        ctrl_f = ttk.Frame(left)
        ctrl_f.pack(fill="x", pady=8)
        self.solve_btn = ttk.Button(ctrl_f, text="SOLVE", command=self.start_solving)
        self.solve_btn.pack(side="left", padx=5)
        self.stop_btn = ttk.Button(ctrl_f, text="STOP", command=self.stop_solving, state="disabled")
        self.stop_btn.pack(side="left", padx=5)

        self.progress = ttk.Progressbar(left, mode='indeterminate')
        self.progress.pack(fill="x", pady=5)

        right = ttk.Frame(main_pane)
        main_pane.add(right, weight=1)

        tabs = ttk.Notebook(right)
        tabs.pack(fill="both", expand=True)

        log_tab = ttk.Frame(tabs)
        tabs.add(log_tab, text="Solutions")
        self.log_text = scrolledtext.ScrolledText(log_tab, font=("Consolas", 10), state='disabled')
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)

        hist_tab = ttk.Frame(tabs)
        tabs.add(hist_tab, text="History")
        self.hist_list = tk.Listbox(hist_tab, font=("Consolas", 10))
        self.hist_list.pack(fill="both", expand=True, padx=5, pady=5)
        self.hist_list.bind("<Double-Button-1>", self.load_history)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w").pack(fill="x", side="bottom")

    def _insert(self, txt):
        e = self.root.focus_get()
        if not isinstance(e, tk.Entry): return
        if txt == 'DEL': e.delete(len(e.get())-1, 'end')
        elif txt == 'CLR': e.delete(0, 'end')
        elif txt == 'π': e.insert('insert', "3.14159265359")
        else: e.insert('insert', txt)

    def log(self, msg):
        self.log_text.configure(state='normal')
        self.log_text.insert('end', msg + '\n')
        self.log_text.see('end')
        self.log_text.configure(state='disabled')

    def analyze_equation(self):
        """Analyze equation and display variable domains"""
        eq = self.equation_var.get().strip()
        if '=' not in eq:
            messagebox.showerror("Error", "Equation must contain '='")
            return

        try:
            left_str, right_str = [s.strip() for s in eq.split('=', 1)]
            left_tree = parse_formula(left_str)
            right_tree = parse_formula(right_str or "0")

            # Extract variables
            variables = {}
            from core.parse import extract_variables
            extract_variables(left_tree.root, variables)
            extract_variables(right_tree.root, variables)

            # Detect domains for each variable
            self.detected_domains = {}
            for var_name in variables:
                domain_min, domain_max = self._detect_domain_for_var(left_tree, right_tree, var_name)
                self.detected_domains[var_name] = (domain_min, domain_max)

            # Update domain UI
            self._update_domain_ui(list(variables.keys()))
            
            self.status_var.set("Equation analyzed successfully")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze equation: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def _detect_domain_for_var(self, left_tree, right_tree, var_name):
        """Detect domain for a specific variable"""
        lower = float('-inf')
        upper = float('inf')

        def walk(node, target_var):
            nonlocal lower, upper
            if isinstance(node, FunctionNode):
                arg = node.argument
                if node.function_name == 'log':
                    # log requires argument > 0
                    if isinstance(arg, BinaryOpNode) and arg.operator == '-':
                        # log(C - var): var < C
                        if isinstance(arg.left, NumberNode) and isinstance(arg.right, VariableNode):
                            if arg.right.name == target_var:
                                upper = min(upper, arg.left.value - 1e-9)
                        # log(var - C): var > C
                        elif isinstance(arg.left, VariableNode) and isinstance(arg.right, NumberNode):
                            if arg.left.name == target_var:
                                lower = max(lower, arg.right.value + 1e-9)
                    elif isinstance(arg, VariableNode) and arg.name == target_var:
                        # log(var): var > 0
                        lower = max(lower, 1e-9)
                    elif isinstance(arg, BinaryOpNode) and arg.operator == '+':
                        # log(var + C): var > -C
                        if isinstance(arg.left, VariableNode) and isinstance(arg.right, NumberNode):
                            if arg.left.name == target_var:
                                lower = max(lower, -arg.right.value + 1e-9)
                
                elif node.function_name == 'sqrt':
                    # sqrt requires argument >= 0
                    if isinstance(arg, BinaryOpNode) and arg.operator == '+':
                        # sqrt(var + C): var >= -C
                        if isinstance(arg.left, VariableNode) and isinstance(arg.right, NumberNode):
                            if arg.left.name == target_var:
                                lower = max(lower, -arg.right.value + 1e-9)
                    elif isinstance(arg, BinaryOpNode) and arg.operator == '-':
                        # sqrt(C - var): var <= C
                        if isinstance(arg.left, NumberNode) and isinstance(arg.right, VariableNode):
                            if arg.right.name == target_var:
                                upper = min(upper, arg.left.value - 1e-9)
                        # sqrt(var - C): var >= C
                        elif isinstance(arg.left, VariableNode) and isinstance(arg.right, NumberNode):
                            if arg.left.name == target_var:
                                lower = max(lower, arg.right.value + 1e-9)
                    elif isinstance(arg, VariableNode) and arg.name == target_var:
                        # sqrt(var): var >= 0
                        lower = max(lower, 1e-9)
                        
                elif node.function_name in ('arcsin', 'arccos'):
                    # arcsin/arccos require -1 <= argument <= 1
                    if isinstance(arg, VariableNode) and arg.name == target_var:
                        lower = max(lower, -1)
                        upper = min(upper, 1)
                        
            elif node.node_type == NodeType.BINARY_OP:
                walk(node.left, target_var)
                walk(node.right, target_var)
            elif node.node_type == NodeType.FUNCTION:
                walk(node.argument, target_var)

        walk(left_tree.root, var_name)
        walk(right_tree.root, var_name)

        # Set default bounds if still infinite
        if lower == float('-inf'):
            lower = -100
        if upper == float('inf'):
            upper = 100

        if lower >= upper:
            # Invalid domain detected, use safe defaults
            lower = -100
            upper = 100

        return lower, upper

    def _update_domain_ui(self, variables):
        """Update the domain configuration UI with detected variables"""
        # Clear existing widgets
        for widget in self.domain_frame.winfo_children():
            widget.destroy()
        
        self.domain_entries = {}

        if not variables:
            ttk.Label(self.domain_frame, text="No variables detected", 
                      foreground="gray").pack(pady=10)
            return

        # Create header
        header_frame = ttk.Frame(self.domain_frame)
        header_frame.pack(fill="x", pady=(0, 5))
        ttk.Label(header_frame, text="Variable", font=("Consolas", 10, "bold")).grid(row=0, column=0, padx=5)
        ttk.Label(header_frame, text="Min", font=("Consolas", 10, "bold")).grid(row=0, column=1, padx=5)
        ttk.Label(header_frame, text="Max", font=("Consolas", 10, "bold")).grid(row=0, column=2, padx=5)
        ttk.Label(header_frame, text="Auto", font=("Consolas", 10, "bold")).grid(row=0, column=3, padx=5)

        # Create entries for each variable
        for i, var_name in enumerate(sorted(variables)):
            frame = ttk.Frame(self.domain_frame)
            frame.pack(fill="x", pady=2)

            # Variable name label
            ttk.Label(frame, text=f"{var_name}:", font=("Consolas", 11)).grid(row=0, column=0, padx=5, sticky="e")

            # Min value entry
            min_var = tk.DoubleVar(value=self.detected_domains[var_name][0])
            min_entry = ttk.Entry(frame, textvariable=min_var, width=15)
            min_entry.grid(row=0, column=1, padx=5)

            # Max value entry
            max_var = tk.DoubleVar(value=self.detected_domains[var_name][1])
            max_entry = ttk.Entry(frame, textvariable=max_var, width=15)
            max_entry.grid(row=0, column=2, padx=5)

            # Auto-detect button
            auto_btn = ttk.Button(frame, text="↻", width=3, 
                                  command=lambda v=var_name, minv=min_var, maxv=max_var: 
                                  self._reset_domain(v, minv, maxv))
            auto_btn.grid(row=0, column=3, padx=5)

            self.domain_entries[var_name] = (min_var, max_var)

        # Add info label
        info_label = ttk.Label(self.domain_frame, 
                               text="Click ↻ to reset to auto-detected values", 
                               font=("Consolas", 9), foreground="gray")
        info_label.pack(pady=(5, 0))

    def _reset_domain(self, var_name, min_var, max_var):
        """Reset domain to auto-detected values"""
        if var_name in self.detected_domains:
            min_var.set(self.detected_domains[var_name][0])
            max_var.set(self.detected_domains[var_name][1])

    def _detect_domain(self, left_tree, right_tree):
        """Legacy method - kept for compatibility"""
        # Just use the first variable's domain
        variables = {}
        from core.parse import extract_variables
        extract_variables(left_tree.root, variables)
        if variables:
            first_var = list(variables.keys())[0]
            return self._detect_domain_for_var(left_tree, right_tree, first_var)
        return -100, 100

    def start_solving(self):
        eq = self.equation_var.get().strip()
        if '=' not in eq:
            messagebox.showerror("Error", "Use '='")
            return

        if not self.domain_entries:
            messagebox.showwarning("Warning", "Please analyze the equation first to set variable domains")
            return

        self.solve_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.progress.start()
        self.solver_thread = threading.Thread(target=self._solve_worker, daemon=True)
        self.solver_thread.start()

    def stop_solving(self):
        if self.solver: self.solver.end_time = 0
        self.status_var.set("Stopping...")

    def _solve_worker(self):
        try:
            self.root.after(0, lambda: self.status_var.set("Parsing..."))
            left_str, right_str = [s.strip() for s in self.equation_var.get().split('=', 1)]
            left_tree = parse_formula(left_str)
            right_tree = parse_formula(right_str or "0")

            solution = create_solution_from_trees(left_tree, right_tree)
            
            # Apply user-defined domains
            for var_name, (min_var, max_var) in self.domain_entries.items():
                if var_name in solution.variables:
                    domain_min = min_var.get()
                    domain_max = max_var.get()
                    solution.variables[var_name].set_domain(domain_min, domain_max)

            self.root.after(0, lambda: self.status_var.set("Solving..."))
            self.solver = GASolver(
                solution_template=solution,
                population_size=self.param_vars[0].get(),
                time_limit=self.param_vars[1].get(),
                tournament_size=self.param_vars[2].get(),
                elitism_size=self.param_vars[3].get()
            )

            def gui_print(*a):
                self.root.after(0, lambda: self.log(' '.join(map(str, a))))
            import builtins
            old_print = builtins.print
            builtins.print = gui_print

            start = time.time()
            solutions = self.solver.solve()
            elapsed = time.time() - start
            builtins.print = old_print

            # Format results for all variables
            result_lines = []
            for i, s in enumerate(solutions, 1):
                var_str = ", ".join([f"{name} = {var.current_value:.10f}" 
                                     for name, var in sorted(s.variables.items())])
                result_lines.append(f"Solution #{i}: {var_str} | Error: {s.calculate_heuristic():.2e}")
            result = "\n".join(result_lines)
            
            self.history.append({"eq": self.equation_var.get(), "result": result, "time": elapsed})
            self.root.after(0, self._update_history)
            self.root.after(0, self._show_results, solutions, elapsed)

        except Exception as e:
            import traceback
            self.root.after(0, lambda: self.log(f"ERROR: {traceback.format_exc()}"))
        finally:
            self.root.after(0, self._reset_ui)

    def _show_results(self, solutions, elapsed):
        self.log("\n" + "="*80)
        self.log(f"SOLVER FINISHED in {elapsed:.2f}s")
        self.log(f"FOUND {len(solutions)} SOLUTION(S):\n")
        for i, s in enumerate(solutions, 1):
            self.log(f" SOLUTION #{i}")
            for var_name, var in sorted(s.variables.items()):
                self.log(f"   {var_name} = {var.current_value:.12f}")
            err = s.calculate_heuristic()
            self.log(f"   Error = {err:.2e}\n")

    def _reset_ui(self):
        self.progress.stop()
        self.solve_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set("Ready")

    def _update_history(self):
        self.hist_list.delete(0, tk.END)
        for h in self.history[-10:]:
            self.hist_list.insert(tk.END, f"{h['eq']} → {len(h['result'].splitlines())} solutions")

    def load_history(self, event):
        sel = self.hist_list.curselection()
        if sel:
            idx = len(self.history) - (10 - sel[0])
            if 0 <= idx < len(self.history):
                self.equation_var.set(self.history[idx]["eq"])
                self.analyze_equation()  # Re-analyze to populate domains


if __name__ == "__main__":
    root = tk.Tk()
    app = SolverGUI(root)
    root.mainloop()