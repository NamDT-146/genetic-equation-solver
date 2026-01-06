import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, List
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.benchmark import BenchmarkGenerator


class ReportGenerator:
    """Generates tables and figures for academic paper"""
    
    def __init__(self, results_file: str):
        with open(results_file, 'r') as f:
            self.data = json.load(f)
        
        self.results = self.data['results']
        self.config = self.data['config']
        
        # Load benchmark metadata
        generator = BenchmarkGenerator()
        self.benchmarks = {b.name: b for b in generator.generate_full_suite()}
        
        # Output directory
        self.output_dir = "evaluation/report_output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Academic plot style
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 9
        plt.rcParams['figure.titlesize'] = 12
    
    def _format_equation_latex(self, equation: str) -> str:
        """Convert equation string to LaTeX math format"""
        # Replace operators
        eq = equation.replace('*', '\\cdot ')
        eq = eq.replace('sqrt(', '\\sqrt{')
        eq = eq.replace('log(', '\\log(')
        eq = eq.replace('sin(', '\\sin(')
        eq = eq.replace('cos(', '\\cos(')
        eq = eq.replace('tan(', '\\tan(')
        eq = eq.replace('arcsin(', '\\arcsin(')
        eq = eq.replace('arccos(', '\\arccos(')
        eq = eq.replace('arctan(', '\\arctan(')
        
        # Handle power operator - need to add closing brace after the exponent
        parts = eq.split('^')
        if len(parts) > 1:
            result = [parts[0]]
            for i in range(1, len(parts)):
                part = parts[i]
                # Find the end of the exponent (next non-digit/non-variable character)
                j = 0
                while j < len(part) and (part[j].isalnum() or part[j] == '.'):
                    j += 1
                exponent = part[:j]
                rest = part[j:]
                result.append('^{' + exponent + '}' + rest)
            eq = ''.join(result)
        
        # Handle nested functions by counting braces
        open_brace = eq.count('{')
        close_brace = eq.count('}')
        
        if open_brace > close_brace:
            # Add closing braces at the end
            eq += '}' * (open_brace - close_brace)
        
        return eq
    
    def generate_benchmark_description_table(self) -> str:
        """Generate LaTeX table describing all benchmark problems"""
        
        # Group by category
        classical = [b for b in self.benchmarks.values() if b.category == 'classical']
        scalable = [b for b in self.benchmarks.values() if b.category == 'scalable']
        
        latex = []
        latex.append("\\begin{table}[H]")
        latex.append("\\centering")
        latex.append("\\caption{Benchmark Problem Suite}")
        latex.append("\\label{tab:benchmark_problems}")
        latex.append("\\resizebox{\\textwidth}{!}{%")
        latex.append("\\begin{tabular}{llccp{6cm}}")
        latex.append("\\toprule")
        latex.append("Category & Problem & Dim & Difficulty & Equation \\\\")
        latex.append("\\midrule")
        
        # Classical problems
        for bench in sorted(classical, key=lambda x: x.name):
            prob_name = bench.name.replace('_', '\\_')
            equation = self._format_equation_latex(bench.equation)
            latex.append(
                f"Classical & {prob_name} & {bench.dimension}D & "
                f"{bench.difficulty.capitalize()} & $\\displaystyle {equation}$ \\\\"
            )
        
        latex.append("\\midrule")
        
        # Scalable problems
        for bench in sorted(scalable, key=lambda x: x.name):
            prob_name = bench.name.replace('_', '\\_')
            equation = self._format_equation_latex(bench.equation)
            # Truncate very long equations
            if len(equation) > 100:
                equation = equation[:97] + "..."
            latex.append(
                f"Scalable & {prob_name} & {bench.dimension}D & "
                f"{bench.difficulty.capitalize()} & $\\displaystyle {equation}$ \\\\"
            )
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}%")
        latex.append("}")
        latex.append("\\end{table}")
        
        latex_str = "\n".join(latex)
        
        # Save to file
        table_file = os.path.join(self.output_dir, "benchmark_problems.tex")
        with open(table_file, 'w') as f:
            f.write(latex_str)
        
        print(f"Benchmark problems table saved to: {table_file}")
        return latex_str
    
    def generate_detailed_results_table(self) -> str:
        """Generate detailed results table with equations"""
        
        # Group results by problem
        problems = {}
        for result in self.results:
            prob_name = result['problem_name']
            if prob_name not in problems:
                problems[prob_name] = []
            problems[prob_name].append(result)
        
        # Compute statistics
        rows = []
        for prob_name in sorted(problems.keys()):
            prob_results = problems[prob_name]
            
            errors = [r['final_error'] for r in prob_results if r['final_error'] != float('inf')]
            times = [r['time_elapsed'] for r in prob_results]
            successes = [r['success'] for r in prob_results]
            
            success_rate = np.mean(successes) * 100
            mean_error = np.mean(errors) if errors else float('inf')
            std_error = np.std(errors) if errors else 0
            mean_time = np.mean(times)
            
            # Get equation
            equation = ""
            if prob_name in self.benchmarks:
                equation = self.benchmarks[prob_name].equation
            
            rows.append({
                'problem': prob_name,
                'equation': equation,
                'success_rate': success_rate,
                'mean_error': mean_error,
                'std_error': std_error,
                'mean_time': mean_time
            })
        
        # Generate LaTeX table
        latex = []
        latex.append("\\begin{table}[H]")
        latex.append("\\centering")
        latex.append("\\caption{Detailed Benchmark Performance Results}")
        latex.append("\\label{tab:detailed_results}")
        latex.append("\\resizebox{\\textwidth}{!}{%")
        latex.append("\\begin{tabular}{lp{4cm}cccc}")
        latex.append("\\toprule")
        latex.append("Problem & Equation & Success Rate & Mean Error & Std Error & Mean Time (s) \\\\")
        latex.append("\\midrule")
        
        for row in rows:
            prob_name = row['problem'].replace('_', '\\_')
            equation = self._format_equation_latex(row['equation'])
            if len(equation) > 50:
                equation = equation[:47] + "..."
            
            latex.append(
                f"{prob_name} & $\\scriptstyle {equation}$ & "
                f"{row['success_rate']:.1f}\\% & "
                f"{row['mean_error']:.2e} & "
                f"{row['std_error']:.2e} & "
                f"{row['mean_time']:.1f} \\\\"
            )
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}%")
        latex.append("}")
        latex.append("\\end{table}")
        
        latex_str = "\n".join(latex)
        
        # Save to file
        table_file = os.path.join(self.output_dir, "detailed_results.tex")
        with open(table_file, 'w') as f:
            f.write(latex_str)
        
        print(f"Detailed results table saved to: {table_file}")
        return latex_str
    
    def generate_summary_table(self) -> str:
        """Generate LaTeX table of summary statistics with equations"""
        
        # Group results by problem
        problems = {}
        for result in self.results:
            prob_name = result['problem_name']
            if prob_name not in problems:
                problems[prob_name] = []
            problems[prob_name].append(result)
        
        # Compute statistics
        rows = []
        for prob_name in sorted(problems.keys()):
            prob_results = problems[prob_name]
            
            errors = [r['final_error'] for r in prob_results if r['final_error'] != float('inf')]
            times = [r['time_elapsed'] for r in prob_results]
            successes = [r['success'] for r in prob_results]
            
            success_rate = np.mean(successes) * 100
            mean_error = np.mean(errors) if errors else float('inf')
            std_error = np.std(errors) if errors else 0
            mean_time = np.mean(times)
            
            # Get equation
            equation = ""
            if prob_name in self.benchmarks:
                equation = self.benchmarks[prob_name].equation
            
            rows.append({
                'problem': prob_name,
                'equation': equation,
                'success_rate': success_rate,
                'mean_error': mean_error,
                'std_error': std_error,
                'mean_time': mean_time
            })
        
        # Generate LaTeX table
        latex = []
        latex.append("\\begin{table}[H]")
        latex.append("\\centering")
        latex.append("\\caption{Benchmark Performance Summary}")
        latex.append("\\label{tab:benchmark_summary}")
        latex.append("\\resizebox{\\textwidth}{!}{%")
        latex.append("\\begin{tabular}{lp{5cm}cccc}")
        latex.append("\\toprule")
        latex.append("Problem & Equation & Success Rate & Mean Error & Std Error & Mean Time (s) \\\\")
        latex.append("\\midrule")
        
        for row in rows:
            prob_name = row['problem'].replace('_', '\\_')
            equation = self._format_equation_latex(row['equation'])
            
            # Shorten equation if too long
            if len(equation) > 60:
                equation = equation[:57] + "..."
            
            latex.append(
                f"{prob_name} & $\\scriptstyle {equation}$ & "
                f"{row['success_rate']:.1f}\\% & "
                f"{row['mean_error']:.2e} & "
                f"{row['std_error']:.2e} & "
                f"{row['mean_time']:.1f} \\\\"
            )
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}%")
        latex.append("}")
        latex.append("\\end{table}")
        
        latex_str = "\n".join(latex)
        
        # Save to file
        table_file = os.path.join(self.output_dir, "summary_table.tex")
        with open(table_file, 'w') as f:
            f.write(latex_str)
        
        print(f"Summary table saved to: {table_file}")
        return latex_str
    
    def generate_category_comparison(self):
        """Generate bar chart comparing classical vs scalable"""
        
        # Group by category
        classical_results = [r for r in self.results if 'Broyden' not in r['problem_name'] and 'Sum_of' not in r['problem_name']]
        scalable_results = [r for r in self.results if 'Broyden' in r['problem_name'] or 'Sum_of' in r['problem_name']]
        
        def compute_stats(results):
            successes = [r['success'] for r in results]
            errors = [r['final_error'] for r in results if r['final_error'] != float('inf')]
            times = [r['time_elapsed'] for r in results]
            
            return {
                'success_rate': np.mean(successes) * 100 if successes else 0,
                'mean_error': np.mean(errors) if errors else float('inf'),
                'mean_time': np.mean(times) if times else 0
            }
        
        classical_stats = compute_stats(classical_results)
        scalable_stats = compute_stats(scalable_results)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        
        categories = ['Classical', 'Scalable']
        
        # Success rate
        axes[0].bar(categories, 
                    [classical_stats['success_rate'], scalable_stats['success_rate']])
        axes[0].set_ylabel('Success Rate (%)')
        axes[0].set_ylim([0, 100])
        axes[0].grid(axis='y', alpha=0.3)
        
        # Mean error (log scale)
        axes[1].bar(categories,
                    [classical_stats['mean_error'], scalable_stats['mean_error']])
        axes[1].set_ylabel('Mean Error')
        axes[1].set_yscale('log')
        axes[1].grid(axis='y', alpha=0.3)
        
        # Mean time
        axes[2].bar(categories,
                    [classical_stats['mean_time'], scalable_stats['mean_time']])
        axes[2].set_ylabel('Mean Time (s)')
        axes[2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        fig_file = os.path.join(self.output_dir, "category_comparison.pdf")
        plt.savefig(fig_file, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Category comparison saved to: {fig_file}")
    
    def generate_dimension_scalability(self):
        """Generate plot showing performance vs dimension"""
        
        # Extract dimensional problems
        dim_problems = {}
        for result in self.results:
            if 'Broyden' in result['problem_name'] or 'Sum_of' in result['problem_name']:
                # Extract dimension from name
                if '2D' in result['problem_name']:
                    dim = 2
                elif '3D' in result['problem_name']:
                    dim = 3
                elif '4D' in result['problem_name']:
                    dim = 4
                elif '5D' in result['problem_name']:
                    dim = 5
                else:
                    continue
                
                if dim not in dim_problems:
                    dim_problems[dim] = []
                dim_problems[dim].append(result)
        
        if not dim_problems:
            print("No dimensional scaling data found")
            return
        
        # Compute statistics per dimension
        dimensions = sorted(dim_problems.keys())
        success_rates = []
        mean_errors = []
        mean_times = []
        
        for dim in dimensions:
            results = dim_problems[dim]
            successes = [r['success'] for r in results]
            errors = [r['final_error'] for r in results if r['final_error'] != float('inf')]
            times = [r['time_elapsed'] for r in results]
            
            success_rates.append(np.mean(successes) * 100)
            mean_errors.append(np.mean(errors) if errors else float('inf'))
            mean_times.append(np.mean(times))
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        
        # Success rate vs dimension
        axes[0].plot(dimensions, success_rates, 'o-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Problem Dimension')
        axes[0].set_ylabel('Success Rate (%)')
        axes[0].set_ylim([0, 100])
        axes[0].grid(alpha=0.3)
        
        # Error vs dimension
        axes[1].plot(dimensions, mean_errors, 'o-', linewidth=2, markersize=8)
        axes[1].set_xlabel('Problem Dimension')
        axes[1].set_ylabel('Mean Error')
        axes[1].set_yscale('log')
        axes[1].grid(alpha=0.3)
        
        # Time vs dimension
        axes[2].plot(dimensions, mean_times, 'o-', linewidth=2, markersize=8)
        axes[2].set_xlabel('Problem Dimension')
        axes[2].set_ylabel('Mean Time (s)')
        axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        
        fig_file = os.path.join(self.output_dir, "dimension_scalability.pdf")
        plt.savefig(fig_file, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Dimension scalability plot saved to: {fig_file}")
    
    def generate_error_distribution(self):
        """Generate histogram of error distribution"""
        
        errors = [r['final_error'] for r in self.results 
                 if r['final_error'] != float('inf') and r['final_error'] > 0]
        
        if not errors:
            print("No valid error data found")
            return
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        ax.hist(np.log10(errors), bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Log10(Error)')
        ax.set_ylabel('Frequency')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        fig_file = os.path.join(self.output_dir, "error_distribution.pdf")
        plt.savefig(fig_file, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Error distribution saved to: {fig_file}")
    
    def generate_full_report(self):
        """Generate all report materials"""
        print("\nGenerating report materials...")
        print("="*50)
        
        self.generate_benchmark_description_table()
        self.generate_detailed_results_table()
        self.generate_summary_table()
        self.generate_category_comparison()
        self.generate_dimension_scalability()
        self.generate_error_distribution()
        
        print("="*50)
        print(f"All report materials saved to: {self.output_dir}")
        print("\nGenerated files:")
        print("  - benchmark_problems.tex (Problem descriptions with equations)")
        print("  - detailed_results.tex (Results with equations)")
        print("  - summary_table.tex (Summary with equations)")
        print("  - category_comparison.pdf")
        print("  - dimension_scalability.pdf")
        print("  - error_distribution.pdf")


def main():
    """Main execution"""
    checkpoint_dir = "evaluation/checkpoints"
    results_file = os.path.join(checkpoint_dir, "latest.json")
    
    if not os.path.exists(results_file):
        print(f"Error: Results file not found: {results_file}")
        print("Please run perform_test.py first to generate results")
        return
    
    generator = ReportGenerator(results_file)
    generator.generate_full_report()


if __name__ == "__main__":
    main()