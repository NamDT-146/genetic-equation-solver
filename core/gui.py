import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import time
import os
import sys

# --- LOGO SETUP ---
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load icon
icon_path = os.path.join(project_root, "icon.ico")  # .ico for Windows
if os.path.exists(icon_path):
    pass  # Will be used
else:
    icon_path = os.path.join(project_root, "icon.png")  # Fallback to .png

from parse import create_solution_from_trees, parse_formula
from ga_solver import GASolver
from parser.string_to_tree import NodeType, FunctionNode, BinaryOpNode, NumberNode, VariableNode


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

        btn_f = ttk.LabelFrame(left, text="Functions", padding=8)
        btn_f.pack(fill="x", pady=5)

        buttons = [
            ('sin(', 'cos(', 'tan(', 'log('),
            ('arcsin(', 'arccos(', 'arctan(', 'sqrt('),
            ('^', '(', ')', 'π'),
            ('x', 'DEL', 'CLR'),
            ('7', '8', '9', '/'),
            ('4', '5', '6', '*'),
            ('1', '2', '3', '-'),
            ('0', '.', '=', '+')
        ]

        for r, row in enumerate(buttons):
            for c, txt in enumerate(row):
                bg = '#90ee90' if txt == 'x' else '#ff6b6b' if txt in ('DEL', 'CLR') else '#e0e0e0'
                cmd = lambda t=txt: self._insert(t)
                b = tk.Button(btn_f, text=txt, bg=bg, width=6, height=1, command=cmd)
                b.grid(row=r, column=c, padx=1, pady=1, sticky="nsew")
            btn_f.grid_columnconfigure(tuple(range(4)), weight=1)

        param_f = ttk.LabelFrame(left, text="GA Settings", padding=10)
        param_f.pack(fill="x", pady=5)

        labels = ["Population:", "Time (s):", "Tournament:", "Elitism:"]
        defaults = [400, 60, 5, 5]
        self.param_vars = [tk.IntVar(value=v) for v in defaults]

        for i, (lbl, var) in enumerate(zip(labels, self.param_vars)):
            ttk.Label(param_f, text=lbl).grid(row=i//2, column=(i%2)*2, sticky="w")
            spin = ttk.Spinbox(param_f, from_=1, to=9999, textvariable=var, width=10)
            spin.grid(row=i//2, column=(i%2)*2 + 1, padx=5, pady=2)

        self.domain_label = ttk.Label(left, text="Domain: x ∈ [auto]", font=("Consolas", 11))
        self.domain_label.pack(pady=5)

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

    def _detect_domain(self, left_tree, right_tree):
        lower = float('-inf')
        upper = float('inf')

        def walk(node):
            nonlocal lower, upper
            if isinstance(node, FunctionNode):
                arg = node.argument
                if node.function_name == 'log':
                    if isinstance(arg, BinaryOpNode) and arg.operator == '-':
                        if (isinstance(arg.left, NumberNode) and isinstance(arg.right, VariableNode)):
                            upper = min(upper, arg.left.value - 1e-9)
                elif node.function_name == 'sqrt':
                    if isinstance(arg, BinaryOpNode) and arg.operator == '+':
                        if (isinstance(arg.left, VariableNode) and isinstance(arg.right, NumberNode)):
                            lower = max(lower, -arg.right.value + 1e-9)
            elif node.node_type == NodeType.BINARY_OP:
                walk(node.left)
                walk(node.right)

        walk(left_tree.root)
        walk(right_tree.root)

        if lower >= upper:
            raise ValueError("No valid domain")

        return lower, upper

    def start_solving(self):
        eq = self.equation_var.get().strip()
        if '=' not in eq:
            messagebox.showerror("Error", "Use '='")
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

            domain_min, domain_max = self._detect_domain(left_tree, right_tree)
            self.root.after(0, lambda: self.domain_label.config(
                text=f"Domain: x ∈ ({domain_min:.6f}, {domain_max:.6f})"))

            solution = create_solution_from_trees(left_tree, right_tree)
            solution.variables['x'].set_domain(domain_min, domain_max)

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

            result = "\n".join([f"x = {s.variables['x'].current_value:.10f} | Error: {s.calculate_heuristic():.2e}"
                                for s in solutions])
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
            x = s.variables['x'].current_value
            err = s.calculate_heuristic()
            self.log(f" SOLUTION #{i}")
            self.log(f"   x = {x:.12f}")
            self.log(f"   Error = {err:.2e}\n")

    def _reset_ui(self):
        self.progress.stop()
        self.solve_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set("Ready")

    def _update_history(self):
        self.hist_list.delete(0, tk.END)
        for h in self.history[-10:]:
            self.hist_list.insert(tk.END, f"{h['eq']} → {len(h['result'].splitlines())} lines")

    def load_history(self, event):
        sel = self.hist_list.curselection()
        if sel:
            idx = self.history.index(self.history[-(10 - sel[0])])
            self.equation_var.set(self.history[idx]["eq"])


if __name__ == "__main__":
    root = tk.Tk()
    app = SolverGUI(root)
    root.mainloop()