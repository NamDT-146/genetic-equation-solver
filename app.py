import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import time
import os
import sys
from queue import Queue, Empty
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

icon_path = os.path.join(project_root, "icon.ico")
if os.path.exists(icon_path):
    pass
else:
    icon_path = os.path.join(project_root, "icon.png")

from core.parse import create_solution_from_trees, parse_formula
from core.ga_solver import GASolver
from core.parser.string_to_tree import NodeType, FunctionNode, BinaryOpNode, NumberNode, VariableNode


class SolverGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GA Equation Solver")

        if os.path.exists(icon_path):
            self.root.iconbitmap(icon_path) if icon_path.endswith(".ico") else self.root.iconphoto(True, tk.PhotoImage(file=icon_path))

        self.root.geometry("1400x900")

        self.solver = None
        self.solver_thread = None
        self.visualizer_thread = None
        self.is_solving = False
        self.history = []
        self.domain_entries = {}
        self.detected_domains = {}
        self.visualization_queue = Queue(maxsize=100)
        
        self.viz_var_1 = tk.StringVar()
        self.viz_var_2 = tk.StringVar()
        self.viz_type = tk.StringVar(value="distribution")
        
        self.current_viz_data = None
        self.last_update_time = 0
        self.viz_update_interval = 0.2  
        
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

        progress_frame = ttk.Frame(left)
        progress_frame.pack(fill="x", pady=5)
        self.progress = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress.pack(fill="x", side="left", expand=True)
        self.progress_label = ttk.Label(progress_frame, text="0%", width=6)
        self.progress_label.pack(side="left", padx=5)

        right = ttk.Frame(main_pane)
        main_pane.add(right, weight=2)

        tabs = ttk.Notebook(right)
        tabs.pack(fill="both", expand=True)

        log_tab = ttk.Frame(tabs)
        tabs.add(log_tab, text="Solutions")
        self.log_text = scrolledtext.ScrolledText(log_tab, font=("Consolas", 10), state='disabled')
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)

        viz_tab = ttk.Frame(tabs)
        tabs.add(viz_tab, text="Visualization")
        self._build_visualization_tab(viz_tab)

        hist_tab = ttk.Frame(tabs)
        tabs.add(hist_tab, text="History")
        self.hist_list = tk.Listbox(hist_tab, font=("Consolas", 10))
        self.hist_list.pack(fill="both", expand=True, padx=5, pady=5)
        self.hist_list.bind("<Double-Button-1>", self.load_history)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w").pack(fill="x", side="bottom")

    def _build_visualization_tab(self, parent):
        """Build the visualization tab with plots"""
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(control_frame, text="Variable 1:").pack(side="left", padx=5)
        self.viz_var_1_combo = ttk.Combobox(control_frame, textvariable=self.viz_var_1, width=10, state="readonly")
        self.viz_var_1_combo.pack(side="left", padx=5)
        
        ttk.Label(control_frame, text="Variable 2:").pack(side="left", padx=5)
        self.viz_var_2_combo = ttk.Combobox(control_frame, textvariable=self.viz_var_2, width=10, state="readonly")
        self.viz_var_2_combo.pack(side="left", padx=5)
        
        ttk.Label(control_frame, text="Type:").pack(side="left", padx=5)
        viz_type_combo = ttk.Combobox(control_frame, textvariable=self.viz_type, 
                                      values=["distribution", "heatmap"], width=12, state="readonly")
        viz_type_combo.pack(side="left", padx=5)
        
        self.viz_var_1_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_visualization())
        self.viz_var_2_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_visualization())
        viz_type_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_visualization())
        
        self.fig = Figure(figsize=(12, 8), dpi=100)
        
        self.ax_convergence = self.fig.add_subplot(2, 2, (1, 2))
        self.ax_convergence.set_title("Fitness Convergence")
        self.ax_convergence.set_xlabel("Generation")
        self.ax_convergence.set_ylabel("Fitness (log scale)")
        self.ax_convergence.set_yscale('log')
        self.ax_convergence.grid(True, alpha=0.3)
        
        self.ax_var1 = self.fig.add_subplot(2, 2, 3)
        self.ax_var2 = self.fig.add_subplot(2, 2, 4)
        
        self.fig.tight_layout(pad=3.0)
        
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

    def _visualizer_worker(self):
        """Separate thread for handling visualization updates with throttling"""
        
        while self.is_solving or not self.visualization_queue.empty():
            try:
                data = None
                data_count = 0
                
                while True:
                    try:
                        data = self.visualization_queue.get_nowait()
                        data_count += 1
                    except Empty:
                        break
                
                if data:
                    self.current_viz_data = data
                    
                    current_time = time.time()
                    if current_time - self.last_update_time >= self.viz_update_interval:
                        self.root.after(0, self._update_visualization_ui, data)
                        self.last_update_time = current_time
                
                time.sleep(0.05)
                
            except Exception as e:
                import traceback
                traceback.print_exc()
        
        if self.current_viz_data:
            self.root.after(0, self._update_visualization_ui, self.current_viz_data)
        
    def _refresh_visualization(self):
        if self.current_viz_data:
            self._update_visualization_ui(self.current_viz_data)

    def _update_visualization_ui(self, data):
        try:
            progress = data['progress']
            self.progress['value'] = progress
            self.progress_label.config(text=f"{progress:.0f}%")
            
            self.ax_convergence.clear()
            self.ax_convergence.set_title(
                f"Fitness Convergence\n(Gen: {data['generation']}, Solutions: {data['graduated_count']})",
                fontsize=11, fontweight='bold'
            )
            self.ax_convergence.set_xlabel("Generation")
            self.ax_convergence.set_ylabel("Fitness (log scale)")
            self.ax_convergence.set_yscale('log')
            self.ax_convergence.grid(True, alpha=0.3)
            
            if data['best_fitness_history'] and len(data['best_fitness_history']) > 0:
                generations = list(range(len(data['best_fitness_history'])))
                
                self.ax_convergence.plot(generations, data['best_fitness_history'], 
                                        'g-', label='Best', linewidth=2.5, marker='o', 
                                        markersize=3, markevery=max(1, len(generations)//20))
                self.ax_convergence.plot(generations, data['avg_fitness_history'], 
                                        'b-', label='Average', linewidth=1.8, alpha=0.8)
                self.ax_convergence.plot(generations, data['worst_fitness_history'], 
                                        'r-', label='Worst', linewidth=1.2, alpha=0.5)
                
                if len(generations) > 0:
                    current_gen = generations[-1]
                    current_best = data['best_fitness_history'][-1]
                    self.ax_convergence.plot(current_gen, current_best, 'go', 
                                            markersize=8, markeredgecolor='darkgreen', 
                                            markeredgewidth=2)
                
                self.ax_convergence.legend(loc='upper right')
                
                if len(generations) > 0:
                    last_best = data['best_fitness_history'][-1]
                    last_avg = data['avg_fitness_history'][-1]
                    self.ax_convergence.text(0.02, 0.98, 
                                           f"Best: {last_best:.2e}\nAvg: {last_avg:.2e}",
                                           transform=self.ax_convergence.transAxes,
                                           verticalalignment='top',
                                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                                           fontsize=9)
            
            population_data = data['population_data']
            var_names = list(population_data.keys())
            
            if var_names:
                var1 = self.viz_var_1.get() if self.viz_var_1.get() in var_names else var_names[0]
                var2 = self.viz_var_2.get() if self.viz_var_2.get() in var_names else (var_names[1] if len(var_names) > 1 else var_names[0])
                
                if var1:
                    self._plot_variable_distribution(self.ax_var1, population_data[var1], var1)
                
                if var2:
                    if self.viz_type.get() == "heatmap" and var1 != var2 and var1 in var_names and var2 in var_names:
                        self.ax_var2.clear()
                        self.ax_var2.set_title(f"{var1} vs {var2} Population Heatmap", 
                                              fontsize=11, fontweight='bold')
                        self.ax_var2.set_xlabel(var1)
                        self.ax_var2.set_ylabel(var2)
                        
                        x_data = population_data[var1]
                        y_data = population_data[var2]
                        
                        if len(x_data) == len(y_data) and len(x_data) > 0:
                            scatter = self.ax_var2.scatter(x_data, y_data, 
                                                          c=range(len(x_data)), 
                                                          cmap='viridis',
                                                          alpha=0.6, s=30, 
                                                          edgecolors='black', linewidths=0.5)
                            self.fig.colorbar(scatter, ax=self.ax_var2, label='Individual Index')
                        self.ax_var2.grid(True, alpha=0.3)
                    else:
                        self._plot_variable_distribution(self.ax_var2, population_data[var2], var2)
            
            self.fig.tight_layout(pad=3.0)
            self.canvas.draw_idle() 
            
        except Exception as e:
            import traceback
            traceback.print_exc()

    def _plot_variable_distribution(self, ax, data, var_name):
        """Plot distribution for a single variable"""
        ax.clear()
        ax.set_title(f"{var_name} Population Distribution", fontsize=11, fontweight='bold')
        ax.set_xlabel(var_name)
        ax.set_ylabel("Density")
        
        if data and len(data) > 0:
            if self.viz_type.get() == "distribution":
                n, bins, patches = ax.hist(data, bins=30, density=True, 
                                          alpha=0.6, color='skyblue', 
                                          edgecolor='black', linewidth=0.8)
                
                cm = plt.get_cmap('RdYlGn_r')
                bin_centers = 0.5 * (bins[:-1] + bins[1:])
                col = bin_centers - min(bin_centers)
                col /= max(col)
                for c, p in zip(col, patches):
                    plt.setp(p, 'facecolor', cm(c))
                
                if len(data) > 5:
                    try:
                        from scipy import stats
                        density = stats.gaussian_kde(data)
                        xs = np.linspace(min(data), max(data), 200)
                        ax.plot(xs, density(xs), 'r-', linewidth=2.5, label='KDE')
                        ax.legend()
                    except:
                        pass  
                
                mean_val = np.mean(data)
                std_val = np.std(data)
                ax.axvline(mean_val, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
                ax.text(0.98, 0.98, f"μ={mean_val:.3f}\nσ={std_val:.3f}\nn={len(data)}",
                       transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=9)
            else:
                y_vals = np.random.normal(0, 0.02, len(data)) 
                ax.scatter(data, y_vals, alpha=0.6, s=30, c='blue', edgecolors='black', linewidths=0.5)
                ax.set_ylim(-0.1, 0.1)
                ax.set_ylabel("Jittered Y")
        
        ax.grid(True, alpha=0.3)

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

            variables = {}
            from core.parse import extract_variables
            extract_variables(left_tree.root, variables)
            extract_variables(right_tree.root, variables)

            self.detected_domains = {}
            for var_name in variables:
                domain_min, domain_max = self._detect_domain_for_var(left_tree, right_tree, var_name)
                self.detected_domains[var_name] = (domain_min, domain_max)

            self._update_domain_ui(list(variables.keys()))
            
            var_list = sorted(variables.keys())
            self.viz_var_1_combo['values'] = var_list
            self.viz_var_2_combo['values'] = var_list
            if var_list:
                self.viz_var_1.set(var_list[0])
                if len(var_list) > 1:
                    self.viz_var_2.set(var_list[1])
                else:
                    self.viz_var_2.set(var_list[0])
            
            self.status_var.set("Equation analyzed successfully")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze equation: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def _detect_domain_for_var(self, left_tree, right_tree, var_name):
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
                        
            elif node.node_type == NodeType.BINARY_OP:
                walk(node.left, target_var)
                walk(node.right, target_var)
            elif node.node_type == NodeType.FUNCTION:
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

    def _update_domain_ui(self, variables):
        for widget in self.domain_frame.winfo_children():
            widget.destroy()
        
        self.domain_entries = {}

        if not variables:
            ttk.Label(self.domain_frame, text="No variables detected", 
                      foreground="gray").pack(pady=10)
            return

        header_frame = ttk.Frame(self.domain_frame)
        header_frame.pack(fill="x", pady=(0, 5))
        ttk.Label(header_frame, text="Variable", font=("Consolas", 10, "bold")).grid(row=0, column=0, padx=5)
        ttk.Label(header_frame, text="Min", font=("Consolas", 10, "bold")).grid(row=0, column=1, padx=5)
        ttk.Label(header_frame, text="Max", font=("Consolas", 10, "bold")).grid(row=0, column=2, padx=5)
        ttk.Label(header_frame, text="Auto", font=("Consolas", 10, "bold")).grid(row=0, column=3, padx=5)

        for i, var_name in enumerate(sorted(variables)):
            frame = ttk.Frame(self.domain_frame)
            frame.pack(fill="x", pady=2)

            ttk.Label(frame, text=f"{var_name}:", font=("Consolas", 11)).grid(row=0, column=0, padx=5, sticky="e")

            min_var = tk.DoubleVar(value=self.detected_domains[var_name][0])
            min_entry = ttk.Entry(frame, textvariable=min_var, width=15)
            min_entry.grid(row=0, column=1, padx=5)

            max_var = tk.DoubleVar(value=self.detected_domains[var_name][1])
            max_entry = ttk.Entry(frame, textvariable=max_var, width=15)
            max_entry.grid(row=0, column=2, padx=5)

            auto_btn = ttk.Button(frame, text="↻", width=3, 
                                  command=lambda v=var_name, minv=min_var, maxv=max_var: 
                                  self._reset_domain(v, minv, maxv))
            auto_btn.grid(row=0, column=3, padx=5)

            self.domain_entries[var_name] = (min_var, max_var)

        info_label = ttk.Label(self.domain_frame, 
                               text="Click ↻ to reset to auto-detected values", 
                               font=("Consolas", 9), foreground="gray")
        info_label.pack(pady=(5, 0))

    def _reset_domain(self, var_name, min_var, max_var):
        if var_name in self.detected_domains:
            min_var.set(self.detected_domains[var_name][0])
            max_var.set(self.detected_domains[var_name][1])

    def start_solving(self):
        eq = self.equation_var.get().strip()
        if '=' not in eq:
            messagebox.showerror("Error", "Use '='")
            return

        if not self.domain_entries:
            messagebox.showwarning("Warning", "Please analyze the equation first to set variable domains")
            return

        while not self.visualization_queue.empty():
            try:
                self.visualization_queue.get_nowait()
            except:
                break

        self.progress['value'] = 0
        self.progress_label.config(text="0%")
        self.current_viz_data = None
        self.last_update_time = 0

        self.is_solving = True

        self.solve_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        
        self.solver_thread = threading.Thread(target=self._solve_worker, daemon=True)
        self.solver_thread.start()
        
        self.visualizer_thread = threading.Thread(target=self._visualizer_worker, daemon=True)
        self.visualizer_thread.start()

    def stop_solving(self):
        if self.solver: 
            self.solver.end_time = 0
        self.is_solving = False
        self.status_var.set("Stopping...")

    def _solve_worker(self):
        try:
            self.root.after(0, lambda: self.status_var.set("Parsing..."))
            left_str, right_str = [s.strip() for s in self.equation_var.get().split('=', 1)]
            left_tree = parse_formula(left_str)
            right_tree = parse_formula(right_str or "0")

            solution = create_solution_from_trees(left_tree, right_tree)
            
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
                elitism_size=self.param_vars[3].get(),
                visualization_queue=self.visualization_queue,
                visualization_interval=2  # Push every 2 generations
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

            # Mark solving as complete
            self.is_solving = False

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
            self.is_solving = False
            self.root.after(0, lambda: self.log(f"ERROR: {traceback.format_exc()}"))
        finally:
            self.is_solving = False
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
        self.progress['value'] = 100
        self.progress_label.config(text="100%")
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
                self.analyze_equation()


if __name__ == "__main__":
    import matplotlib.pyplot as plt 
    root = tk.Tk()
    app = SolverGUI(root)
    root.mainloop()