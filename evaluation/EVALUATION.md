# Evaluation Methodology

## Overview

This evaluation framework provides a systematic assessment of the Genetic Algorithm (GA) solver for equation solving. The methodology follows standard practices from numerical analysis and evolutionary computation literature.

## Benchmark Suite

### Design Principles

The benchmark suite is divided into two tiers to test distinct aspects of the solver:

1. **Classical Tier**: Tests accuracy and reliability on well-known problems
2. **Scalable Tier**: Tests dimensional scalability and computational efficiency

### Tier A: Classical Nonlinear Systems

These problems are derived from numerical analysis literature (Moré, Garbow, and Hillstrom, 1981) and serve as validation benchmarks.

**Problems included:**
- Quadratic equations (1D)
- Cubic polynomials (1D)
- Transcendental equations (1D, 2D)
- Rosenbrock system (2D)
- Powell's singular function (2D)

**Purpose:** Verify that the GA can solve standard problems that gradient-based methods handle well.

### Tier B: Scalable Systems

These problems test how the solver handles increasing dimensionality (curse of dimensionality).

**Problems included:**
- Broyden Tridiagonal System (2D, 3D, 5D)
- Sum of Squares System (2D, 3D, 4D)

**Purpose:** Assess computational scalability and solution quality degradation with dimension.

## Evaluation Metrics

### Primary Metrics

1. **Success Rate**: Percentage of runs achieving error < 1e-6
2. **Mean Error**: Average residual |f(x)| across all runs
3. **Mean Time**: Average computational time per run

### Statistical Rigor

- **Number of runs**: 10 independent runs per problem (reduced from 30 for practical testing)
- **Reporting**: Mean ± Standard Deviation
- **Success criterion**: Residual < 1e-6 (standard for numerical solvers)

## Implementation

### Benchmark Generator (`benchmark.py`)

**Class: `BenchmarkProblem`**
- Encapsulates problem metadata (name, equation, dimension, known solutions)
- Automatic domain detection using constraint analysis
- Conversion to `Solution` object for GA solver

**Class: `BenchmarkGenerator`**
- Generates complete benchmark suites
- Two methods: `generate_classical_suite()`, `generate_scalable_suite()`
- Extensible design for adding new problems

### Test Performer (`perform_test.py`)

**Class: `TestConfig`**
- Centralized configuration for GA parameters
- Checkpoint settings for long-running tests

**Class: `BenchmarkTester`**
- Executes systematic testing with checkpointing
- Computes summary statistics
- Handles interruptions and resumption

**Key Features:**
- **Checkpointing**: Saves progress after each problem
- **ETA Estimation**: Real-time progress tracking
- **Resume Capability**: Continue from last checkpoint
- **Error Handling**: Robust exception management

### Report Generator (`report.py`)

**Class: `ReportGenerator`**
- Generates publication-ready tables and figures
- Academic plotting style (no titles in figures)
- LaTeX table output

**Outputs:**
1. Summary table (LaTeX format)
2. Category comparison (bar charts)
3. Dimension scalability (line plots)
4. Error distribution (histogram)

## Usage

### 1. Generate and Run Benchmarks

```bash
python evaluation/perform_test.py
```

**Expected output:**
- Progress tracking with ETA
- Checkpoint files in `evaluation/checkpoints/`
- Final results in `latest.json`

**Estimated time**: ~5-10 minutes (depends on `time_limit` and `num_runs`)

### 2. Generate Report Materials

```bash
python evaluation/report.py
```

**Outputs** (in `evaluation/report_output/`):
- `summary_table.tex`: LaTeX table for paper
- `category_comparison.pdf`: Classical vs Scalable comparison
- `dimension_scalability.pdf`: Performance vs dimension
- `error_distribution.pdf`: Error histogram

### 3. Resume Interrupted Tests

If testing is interrupted:

```bash
python evaluation/perform_test.py
# Answer 'y' when prompted to resume
```

## Configuration

### Fast Testing (Development)

Edit `TestConfig` in `perform_test.py`:

```python
self.population_size = 100
self.time_limit = 10  # 10 seconds per problem
self.num_runs = 5     # 5 runs per problem
```

**Estimated time**: ~2-3 minutes

### Full Evaluation (Paper Results)

```python
self.population_size = 400
self.time_limit = 60  # 60 seconds per problem
self.num_runs = 30    # 30 runs per problem
```

**Estimated time**: ~30-45 minutes

## Interpreting Results

### Success Rate
- **>80%**: Excellent performance
- **50-80%**: Good performance
- **<50%**: Requires algorithm tuning

### Mean Error
- **<1e-6**: Achieved target accuracy
- **1e-6 to 1e-3**: Partial convergence
- **>1e-3**: Failed to converge

### Dimension Scalability
- Ideal: Success rate remains constant
- Acceptable: Gradual decrease with dimension
- Poor: Exponential degradation

## References

1. Moré, J. J., Garbow, B. S., & Hillstrom, K. E. (1981). Testing unconstrained optimization software. *ACM Transactions on Mathematical Software*, 7(1), 17-41.

2. Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., & Brockhoff, D. (2021). COCO: A platform for comparing continuous optimizers in a black-box setting. *Optimization Methods and Software*, 36(1), 114-144.

## Troubleshooting

### Issue: Tests taking too long
**Solution**: Reduce `time_limit` or `num_runs` in `TestConfig`

### Issue: Checkpoint not resuming
**Solution**: Check `evaluation/checkpoints/latest.json` exists

### Issue: Low success rates
**Solution**: Increase `population_size` or `time_limit` in `TestConfig`

### Issue: Memory errors
**Solution**: Reduce `population_size` or number of islands in `GASolver`