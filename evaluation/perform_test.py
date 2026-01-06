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


class TestConfig:
    """Configuration for testing"""
    def __init__(self):
        # GA Parameters (reduced for faster testing)
        self.population_size = 200
        self.time_limit = 30  # seconds per problem
        self.tournament_size = 5
        self.elitism_size = 5
        
        # Test Parameters
        self.num_runs = 10  # Reduced from 30 for faster testing
        self.success_threshold = 1e-6
        
        # Checkpoint
        self.checkpoint_dir = "evaluation/checkpoints"
        self.checkpoint_interval = 1  # Save after each problem


class TestResult:
    """Stores results for a single run"""
    def __init__(self, problem_name: str, run_id: int):
        self.problem_name = problem_name
        self.run_id = run_id
        self.success = False
        self.final_error = float('inf')
        self.generations = 0
        self.time_elapsed = 0.0
        self.solutions_found = []
        self.best_solution = None
        
    def to_dict(self):
        return {
            'problem_name': self.problem_name,
            'run_id': self.run_id,
            'success': self.success,
            'final_error': self.final_error,
            'generations': self.generations,
            'time_elapsed': self.time_elapsed,
            'solutions_found': len(self.solutions_found),
            'best_solution': self.best_solution
        }


class BenchmarkTester:
    """Performs systematic testing of GA solver on benchmark problems"""
    
    def __init__(self, config: TestConfig = None):
        self.config = config or TestConfig()
        self.results = []
        
        # Create checkpoint directory
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
    def run_single_test(self, benchmark: BenchmarkProblem, run_id: int) -> TestResult:
        """Run GA solver on a single benchmark problem"""
        result = TestResult(benchmark.name, run_id)
        
        try:
            # Convert benchmark to Solution object
            solution_template = benchmark.to_solution()
            
            # Initialize GA Solver
            solver = GASolver(
                solution_template=solution_template,
                population_size=self.config.population_size,
                time_limit=self.config.time_limit,
                tournament_size=self.config.tournament_size,
                elitism_size=self.config.elitism_size,
                eps=self.config.success_threshold
            )
            
            # Run solver
            start_time = time.time()
            solutions = solver.solve()
            result.time_elapsed = time.time() - start_time
            
            # Extract results
            result.generations = solver.generation_count
            result.solutions_found = solutions
            
            if solutions:
                # Find best solution
                best_sol = min(solutions, key=lambda s: s.calculate_heuristic() or float('inf'))
                result.final_error = best_sol.calculate_heuristic() or float('inf')
                result.success = result.final_error < self.config.success_threshold
                
                # Store best solution values
                result.best_solution = {
                    var_name: var.current_value 
                    for var_name, var in best_sol.variables.items()
                }
            
        except Exception as e:
            print(f"Error in test {benchmark.name} run {run_id}: {e}")
            import traceback
            traceback.print_exc()
        
        return result
    
    def run_benchmark_suite(self, benchmarks: List[BenchmarkProblem], 
                           resume_from: str = None):
        """Run full benchmark suite with checkpointing"""
        
        # Load checkpoint if resuming
        completed_tests = set()
        if resume_from and os.path.exists(resume_from):
            with open(resume_from, 'r') as f:
                checkpoint_data = json.load(f)
                self.results = [TestResult(r['problem_name'], r['run_id']) 
                               for r in checkpoint_data['results']]
                for r_dict, r_obj in zip(checkpoint_data['results'], self.results):
                    r_obj.__dict__.update(r_dict)
                
                for r in self.results:
                    completed_tests.add((r.problem_name, r.run_id))
            
            print(f"Resumed from checkpoint: {len(completed_tests)} tests completed")
        
        total_tests = len(benchmarks) * self.config.num_runs
        completed = len(completed_tests)
        
        print(f"\n{'='*70}")
        print(f"BENCHMARK EVALUATION")
        print(f"{'='*70}")
        print(f"Total benchmarks: {len(benchmarks)}")
        print(f"Runs per benchmark: {self.config.num_runs}")
        print(f"Total tests: {total_tests}")
        print(f"Estimated time: {total_tests * self.config.time_limit / 60:.1f} minutes")
        print(f"{'='*70}\n")
        
        # Run tests
        for bench_idx, benchmark in enumerate(benchmarks, 1):
            print(f"\n[{bench_idx}/{len(benchmarks)}] Testing: {benchmark.name}")
            print(f"  Category: {benchmark.category} | Dimension: {benchmark.dimension}D")
            print(f"  Equation: {benchmark.equation}")
            print(f"  Difficulty: {benchmark.difficulty}")
            
            bench_start = time.time()
            
            for run in range(1, self.config.num_runs + 1):
                # Skip if already completed
                if (benchmark.name, run) in completed_tests:
                    print(f"  Run {run}/{self.config.num_runs}: [SKIPPED - already completed]")
                    continue
                
                completed += 1
                eta_seconds = (total_tests - completed) * self.config.time_limit
                eta_minutes = eta_seconds / 60
                
                print(f"  Run {run}/{self.config.num_runs} | Progress: {completed}/{total_tests} | ETA: {eta_minutes:.1f} min", end=" ")
                
                result = self.run_single_test(benchmark, run)
                self.results.append(result)
                
                status = "✓ SUCCESS" if result.success else "✗ FAILED"
                print(f"| {status} | Error: {result.final_error:.2e} | Time: {result.time_elapsed:.1f}s")
            
            bench_time = time.time() - bench_start
            print(f"  Benchmark completed in {bench_time:.1f}s")
            
            # Save checkpoint after each benchmark
            self.save_checkpoint()
        
        print(f"\n{'='*70}")
        print("EVALUATION COMPLETED")
        print(f"{'='*70}\n")
        
        return self.results
    
    def save_checkpoint(self):
        """Save current results to checkpoint file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = os.path.join(
            self.config.checkpoint_dir, 
            f"checkpoint_{timestamp}.json"
        )
        
        checkpoint_data = {
            'timestamp': timestamp,
            'config': {
                'population_size': self.config.population_size,
                'time_limit': self.config.time_limit,
                'num_runs': self.config.num_runs,
                'success_threshold': self.config.success_threshold
            },
            'results': [r.to_dict() for r in self.results]
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        # Also save as "latest" for easy resuming
        latest_file = os.path.join(self.config.checkpoint_dir, "latest.json")
        with open(latest_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    
    def get_summary_statistics(self) -> Dict:
        """Compute summary statistics for report"""
        if not self.results:
            return {}
        
        # Group by problem
        problems = {}
        for result in self.results:
            if result.problem_name not in problems:
                problems[result.problem_name] = []
            problems[result.problem_name].append(result)
        
        # Compute statistics per problem
        summary = {}
        for prob_name, prob_results in problems.items():
            errors = [r.final_error for r in prob_results if r.final_error != float('inf')]
            times = [r.time_elapsed for r in prob_results]
            successes = [r.success for r in prob_results]
            
            summary[prob_name] = {
                'success_rate': np.mean(successes) * 100,
                'mean_error': np.mean(errors) if errors else float('inf'),
                'std_error': np.std(errors) if errors else 0,
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_error': min(errors) if errors else float('inf'),
                'max_error': max(errors) if errors else float('inf')
            }
        
        return summary


def main():
    """Main execution function"""
    
    # Generate benchmarks
    print("Generating benchmark suite...")
    generator = BenchmarkGenerator()
    benchmarks = generator.generate_full_suite()
    
    print(f"Generated {len(benchmarks)} benchmark problems")
    print(f"  Classical: {len([b for b in benchmarks if b.category == 'classical'])}")
    print(f"  Scalable: {len([b for b in benchmarks if b.category == 'scalable'])}")
    
    # Setup test configuration
    config = TestConfig()
    
    # Ask user for resume option
    checkpoint_latest = os.path.join(config.checkpoint_dir, "latest.json")
    resume_from = None
    
    if os.path.exists(checkpoint_latest):
        response = input(f"\nFound checkpoint. Resume from last run? (y/n): ")
        if response.lower() == 'y':
            resume_from = checkpoint_latest
    
    # Run tests
    tester = BenchmarkTester(config)
    results = tester.run_benchmark_suite(benchmarks, resume_from=resume_from)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    summary = tester.get_summary_statistics()
    print(f"\n{'Problem':<30} | {'Success Rate':<12} | {'Mean Error':<15} | {'Mean Time':<10}")
    print("-" * 80)
    
    for prob_name, stats in summary.items():
        print(f"{prob_name:<30} | {stats['success_rate']:>10.1f}% | "
              f"{stats['mean_error']:>15.2e} | {stats['mean_time']:>8.1f}s")
    
    # Save final results
    final_file = os.path.join(config.checkpoint_dir, "final_results.json")
    tester.save_checkpoint()
    
    print(f"\nResults saved to: {final_file}")


if __name__ == "__main__":
    main()