import json
import time
import os
import sys
from datetime import datetime
from typing import List, Dict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.benchmark import BenchmarkGenerator, BenchmarkProblem
from core.ga_solver import GASolver
from core.parse import Solution
from evaluation.perform_test import TestResult


class ExtendedTestConfig:
    """Configuration for extended testing on difficult problems"""
    def __init__(self):
        # GA Parameters (increased for difficult problems)
        self.population_size = 500  # Increased from 200
        self.time_limit = 300  # 5 minutes per run (increased from 30s)
        self.tournament_size = 7
        self.elitism_size = 10
        
        # Test Parameters
        self.num_runs = 30  # Full 30 runs for statistical significance
        self.success_threshold = 1e-6
        
        # Output
        self.output_dir = "evaluation/extended_results"


class ExtendedTester:
    """Extended testing for difficult problems"""
    
    def __init__(self, config: ExtendedTestConfig = None):
        self.config = config or ExtendedTestConfig()
        self.results = []
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def run_single_test(self, benchmark: BenchmarkProblem, run_id: int) -> TestResult:
        """Run GA solver on a single benchmark problem"""
        result = TestResult(benchmark.name, run_id)
        
        try:
            solution_template = benchmark.to_solution()
            
            solver = GASolver(
                solution_template=solution_template,
                population_size=self.config.population_size,
                time_limit=self.config.time_limit,
                tournament_size=self.config.tournament_size,
                elitism_size=self.config.elitism_size,
                eps=self.config.success_threshold
            )
            
            start_time = time.time()
            solutions = solver.solve()
            result.time_elapsed = time.time() - start_time
            
            result.generations = solver.generation_count
            result.solutions_found = solutions
            
            if solutions:
                best_sol = min(solutions, key=lambda s: s.calculate_heuristic() or float('inf'))
                result.final_error = best_sol.calculate_heuristic() or float('inf')
                result.success = result.final_error < self.config.success_threshold
                
                result.best_solution = {
                    var_name: var.current_value 
                    for var_name, var in best_sol.variables.items()
                }
            
        except Exception as e:
            print(f"Error in test {benchmark.name} run {run_id}: {e}")
            import traceback
            traceback.print_exc()
        
        return result
    
    def run_extended_tests(self, problem_names: List[str]):
        """Run extended tests on specific problems"""
        
        # Generate all benchmarks and filter
        generator = BenchmarkGenerator()
        all_benchmarks = generator.generate_full_suite()
        
        benchmarks = [b for b in all_benchmarks if b.name in problem_names]
        
        if not benchmarks:
            print(f"No benchmarks found matching: {problem_names}")
            return
        
        print(f"\n{'='*70}")
        print(f"EXTENDED EVALUATION - DIFFICULT PROBLEMS")
        print(f"{'='*70}")
        print(f"Problems to test: {len(benchmarks)}")
        print(f"Runs per problem: {self.config.num_runs}")
        print(f"Time limit per run: {self.config.time_limit}s ({self.config.time_limit/60:.1f} min)")
        print(f"Population size: {self.config.population_size}")
        print(f"Estimated total time: {len(benchmarks) * self.config.num_runs * self.config.time_limit / 60:.1f} minutes")
        print(f"{'='*70}\n")
        
        for bench_idx, benchmark in enumerate(benchmarks, 1):
            print(f"\n[{bench_idx}/{len(benchmarks)}] Testing: {benchmark.name}")
            print(f"  Category: {benchmark.category} | Dimension: {benchmark.dimension}D")
            print(f"  Equation: {benchmark.equation}")
            print(f"  Difficulty: {benchmark.difficulty}")
            
            bench_results = []
            bench_start = time.time()
            
            for run in range(1, self.config.num_runs + 1):
                print(f"  Run {run}/{self.config.num_runs}", end=" ")
                
                result = self.run_single_test(benchmark, run)
                bench_results.append(result)
                self.results.append(result)
                
                status = "✓ SUCCESS" if result.success else "✗ FAILED"
                print(f"| {status} | Error: {result.final_error:.2e} | "
                      f"Generations: {result.generations} | Time: {result.time_elapsed:.1f}s")
            
            bench_time = time.time() - bench_start
            
            # Print intermediate statistics
            successes = sum(1 for r in bench_results if r.success)
            success_rate = (successes / len(bench_results)) * 100
            errors = [r.final_error for r in bench_results if r.final_error != float('inf')]
            
            print(f"\n  Benchmark Results:")
            print(f"    Success Rate: {success_rate:.1f}% ({successes}/{len(bench_results)})")
            if errors:
                print(f"    Mean Error: {np.mean(errors):.2e}")
                print(f"    Min Error: {min(errors):.2e}")
                print(f"    Max Error: {max(errors):.2e}")
            print(f"    Time: {bench_time:.1f}s ({bench_time/60:.1f} min)")
            
            # Save results after each benchmark
            self.save_results()
        
        print(f"\n{'='*70}")
        print("EXTENDED EVALUATION COMPLETED")
        print(f"{'='*70}\n")
        
        self.print_final_summary()
        
        return self.results
    
    def save_results(self):
        """Save results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            self.config.output_dir,
            f"extended_results_{timestamp}.json"
        )
        
        results_data = {
            'timestamp': timestamp,
            'config': {
                'population_size': self.config.population_size,
                'time_limit': self.config.time_limit,
                'num_runs': self.config.num_runs,
                'success_threshold': self.config.success_threshold
            },
            'results': [r.to_dict() for r in self.results]
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"  Results saved to: {output_file}")
    
    def print_final_summary(self):
        """Print comprehensive summary"""
        if not self.results:
            return
        
        # Group by problem
        problems = {}
        for result in self.results:
            if result.problem_name not in problems:
                problems[result.problem_name] = []
            problems[result.problem_name].append(result)
        
        print("\n" + "="*70)
        print("FINAL SUMMARY STATISTICS")
        print("="*70)
        
        for prob_name, prob_results in problems.items():
            successes = sum(1 for r in prob_results if r.success)
            success_rate = (successes / len(prob_results)) * 100
            errors = [r.final_error for r in prob_results if r.final_error != float('inf')]
            times = [r.time_elapsed for r in prob_results]
            generations = [r.generations for r in prob_results]
            
            print(f"\n{prob_name}:")
            print(f"  Success Rate: {success_rate:.1f}% ({successes}/{len(prob_results)})")
            print(f"  Error Statistics:")
            if errors:
                print(f"    Mean: {np.mean(errors):.2e} ± {np.std(errors):.2e}")
                print(f"    Min:  {min(errors):.2e}")
                print(f"    Max:  {max(errors):.2e}")
            print(f"  Time Statistics:")
            print(f"    Mean: {np.mean(times):.1f}s ± {np.std(times):.1f}s")
            print(f"  Generations:")
            print(f"    Mean: {np.mean(generations):.0f} ± {np.std(generations):.0f}")


def main():
    """Main execution function"""
    
    # Problems with 0% success rate from the table
    difficult_problems = [
        "Broyden_Tridiagonal_5D",
        # "Mixed_Transcendental"
    ]
    
    print("Extended Testing Script")
    print("Problems to test:")
    for prob in difficult_problems:
        print(f"  - {prob}")
    
    # Setup configuration
    config = ExtendedTestConfig()
    
    # Run extended tests
    tester = ExtendedTester(config)
    results = tester.run_extended_tests(difficult_problems)
    
    print(f"\nAll results saved to: {config.output_dir}")


if __name__ == "__main__":
    main()