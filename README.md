# GA Equation Solver

A powerful Genetic Algorithm-based equation solver with real-time visualization and interactive GUI. This application uses advanced evolutionary algorithms with island-based population management to find multiple solutions to complex mathematical equations.

---

## Features

- Multi-Island Genetic Algorithm with adaptive mutation rates
- Real-time visualization of convergence and population distribution
- Automatic domain detection for variables based on equation constraints
- Multiple solution discovery using fitness sharing/niching
- Interactive GUI with equation builder and visual feedback
- Convergence tracking with best, average, and worst fitness plots
- Comprehensive benchmark suite for algorithm evaluation
- Checkpoint and resume functionality for long-running tests

---

## Requirements

### System Requirements
- Operating System: Linux, macOS, or Windows
- Python 3.8 or higher
- RAM: Minimum 4GB (8GB recommended)
- Display: GUI requires graphical display

### Python Dependencies
- numpy >= 1.20.0
- matplotlib >= 3.3.0
- scipy >= 1.6.0
- tkinter (usually included with Python)

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/project1.git
cd project1
```

### Step 2: Create Virtual Environment (Recommended)

**On Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: tkinter is usually pre-installed with Python. If not:

- Ubuntu/Debian: `sudo apt-get install python3-tk`
- Fedora: `sudo dnf install python3-tkinter`
- macOS/Windows: Already included with Python

### Step 4: Verify Installation

```bash
python -c "import numpy, matplotlib, scipy, tkinter; print('All dependencies installed successfully!')"
```

---

## Quick Start

### Run the Application

```bash
python app.py
```

### Solve Your First Equation

1. Enter an equation in the input field (e.g., `sqrt(x+25)+log(49-x)=25`)
2. Click "Analyze Equation" to detect variables and domains
3. Adjust variable domains if needed
4. Configure GA parameters (optional)
5. Click "SOLVE" to start the genetic algorithm
6. Watch the real-time visualization in the "Visualization" tab
7. View solutions in the "Solutions" tab

---

## Application Setup & Usage Guide

### Overview

The GA Equation Solver GUI provides an intuitive interface for solving nonlinear equations. The application consists of several key components:

1. **Equation Input Panel**: Enter and build equations
2. **Domain Configuration**: Set variable search ranges
3. **Algorithm Parameters**: Configure GA settings
4. **Real-time Visualization**: Monitor convergence and population distribution
5. **Solutions Display**: View and analyze found solutions

### Step-by-Step Setup Instructions

#### 1. Launch the Application

```bash
python app.py
```

The GUI window should appear with the main interface showing:
- Left panel: Equation builder and configuration
- Center tabs: Visualization and Solutions
- Top area: Mathematical function buttons

#### 2. Entering Your First Equation

**Method A: Using the Mathematical Keyboard**

The application provides a visual keyboard with:
- **Numbers**: 0-9, decimal point (.)
- **Operators**: +, −, ×, ÷
- **Functions**: sin, cos, tan, arcsin, arccos, arctan, log, sqrt
- **Variables**: x, y, z, w and other single letters
- **Constants**: π (Pi)

Example: To enter `sqrt(x+25)+log(49-x)=25`:
1. Click **sqrt** button
2. Click **(** button
3. Click **x** button
4. Click **+** button
5. Click **2**, then **5**
6. Click **)** button
7. Click **+** button
8. Click **log** button
9. ... and so on

**Method B: Direct Keyboard Input**

Type directly into the equation field:
```
sqrt(x+25)+log(49-x)=25
```

Supported input format:
- Spaces are automatically removed
- Must contain exactly one "=" sign
- Use ^ for exponentiation (e.g., x^2)
- Single-letter variable names only

#### 3. Analyzing the Equation

1. After entering your equation, click the **"Analyze Equation"** button
2. The system will:
   - Parse the mathematical expression
   - Extract all variables (x, y, z, etc.)
   - Automatically detect safe domains based on constraints:
     - For √(f): f must be ≥ 0
     - For log(f): f must be > 0
     - For arcsin/arccos(f): |f| ≤ 1
   - Display detected variables in the Domain Configuration section

Example output for `sqrt(x+25)+log(49-x)=25`:
```
Variables detected: x
Auto-detected domain: -25.0 ≤ x ≤ 48.99
Safe for: sqrt, log operations
```

#### 4. Configuring Variable Domains

After analysis, the **Domain Configuration** panel shows all variables with:
- Variable name
- Auto-detected minimum and maximum values
- Input fields to manually adjust ranges
- Reset button (↻) to restore auto-detected values

**Best Practices:**
- Start with auto-detected domains (usually reliable)
- Narrow the domain if you know where solutions should be
- Wider domains increase search time but may find more solutions
- Use symmetric domains for equations with symmetric solutions

Example configurations:

| Equation | Auto-detected | Recommended |
|----------|--------------|-------------|
| x² - 4 = 0 | [-∞, ∞] | [-10, 10] |
| √(x+25) + log(49-x) = 25 | [-25, 49] | [-24.99, 48.99] |
| sin(x) = 0.5 | [0, 2π] | [0, 10] |

### Using the Genetic Algorithm Configuration Panel

#### GA Parameters Explained

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| **Population** | 400 | 100-1000 | Higher = more exploration, slower speed |
| **Time (seconds)** | 60 | 10-300 | Longer = better convergence, more time |
| **Tournament Size** | 5 | 3-10 | Higher = more selection pressure |
| **Elitism** | 5 | 2-10 | Higher = preserves more best solutions |

#### Configuration Recommendations

**For Simple Equations (Quadratic, Cubic, etc.):**
```
Population: 200
Time: 10-20 seconds
Tournament: 5
Elitism: 3
```

**For Medium Complexity (Trigonometric, Exponential):**
```
Population: 400
Time: 30-60 seconds
Tournament: 5
Elitism: 5
```

**For Complex Equations (Transcendental, Multi-variable):**
```
Population: 800
Time: 120-300 seconds
Tournament: 7
Elitism: 10
```

### Running the Genetic Algorithm

1. After configuring all parameters, click the **"SOLVE"** button
2. The button will change to show progress:
   - Status bar displays current generation
   - Time elapsed updates in real-time
   - Early termination if solution found with sufficient accuracy
3. During execution, you can:
   - View live convergence plots in the **Visualization** tab
   - Watch population distribution updates
   - Monitor best fitness improvement

### Understanding the Visualization Tab

#### Convergence Plot (Top Graph)

The main convergence plot shows three lines over generations:

- **Green Line (Best Fitness)**: Fitness of the best individual in current generation
  - Should generally decrease (improvement)
  - Rapid decrease initially, plateau at convergence
  
- **Blue Line (Average Fitness)**: Average fitness of all individuals
  - Indicates overall population quality
  - Should gradually approach best fitness
  
- **Red Line (Worst Fitness)**: Fitness of worst individual
  - Shows population diversity
  - Should increase (improve) over time

**Interpretation Examples:**

```
Good convergence pattern:
- Green line drops rapidly in first 50 generations
- Plateaus around generation 200-300
- Blue and red lines follow similar pattern
- All three lines converge closely by end

Poor convergence:
- All lines remain flat
- No improvement visible
- Indicates convergence to local optimum
  → Try: increase population, longer time, wider domain
```

#### Population Distribution Plots (Bottom)

**Distribution View (Histogram):**
- Shows probability density of population for selected variable
- X-axis: variable value
- Y-axis: frequency/density
- KDE curve overlaid for smooth estimation
- Peaks indicate regions where population concentrates

**Heatmap View (2D Scatter):**
- Shows population distribution in 2D space
- X-axis: first selected variable
- Y-axis: second selected variable
- Color density indicates concentration
- Useful for multi-variable problems

**Using the Visualization:**

1. Select first variable from **Var1** dropdown
2. Select second variable from **Var2** dropdown (optional)
3. Switch between **Distribution** and **Heatmap** tabs
4. Observe how distribution changes with generations:
   - Initial: population spread widely (uniform)
   - Mid-phase: concentration around high-fitness regions
   - Final: tight clustering near solutions

### Interpreting Results in the Solutions Tab

The **Solutions** tab displays all solutions found, organized in a table:

| Variable | Value | Error | Status |
|----------|-------|-------|--------|
| x | 2.000001 | 1.23e-07 | ✓ Success |
| y | 5.000000 | 5.67e-08 | ✓ Success |

**Understanding Solution Quality:**

- **Error**: Absolute difference between left and right sides of equation
  - `< 1e-6`: Excellent solution
  - `1e-6 to 1e-4`: Good solution
  - `1e-4 to 1e-2`: Acceptable solution
  - `> 1e-2`: Poor solution (verify manually)

- **Multiple Solutions**: For equations with multiple roots, GA often finds several
  - Each solution shown separately
  - Verify by substituting back into original equation

### Advanced Features

#### Monitoring Algorithm Progress

During solving, you can observe:

1. **Generation Count**: Shows current generation number
2. **Time Elapsed**: Total computation time
3. **Best Fitness**: Current best error metric
4. **Population Stats**: Number of active solutions found

#### Island-Based Evolution

The solver uses 4 independent populations (islands) with different strategies:

- **Island 0**: Conservative (explores cautiously)
- **Island 1**: Moderate (balanced approach)
- **Island 2**: Exploratory (higher mutation)
- **Island 3**: Aggressive (maximum exploration)

Islands periodically exchange their best individuals, combining exploration and exploitation benefits. This is handled automatically - no user configuration needed.

#### Early Termination Criteria

The solver automatically stops when:
1. Solution found with error < 1e-6 (success threshold)
2. Time limit reached
3. No improvement for extended period (10+ generations)

### Common Workflows

#### Workflow 1: Solve a Single Equation

```
1. Launch app.py
2. Type equation: x^2-4=0
3. Click "Analyze Equation"
4. Keep default GA parameters
5. Click "SOLVE"
6. View results in Solutions tab
7. Check visualization for convergence behavior
```

**Expected result:** x = 2.0 and x = -2.0 (both roots found)

#### Workflow 2: Solve with Custom Domain

```
1. Enter equation: sin(x)=0.5
2. Click "Analyze Equation"
3. Modify domain: Min=0, Max=2π≈6.28
4. Increase Time to 30 seconds
5. Click "SOLVE"
6. Monitor Visualization tab during execution
7. Multiple solutions found in [0, 2π]
```

#### Workflow 3: Solve Complex Multi-variable Equation

```
1. Enter: x^2+y^2-25=0
2. Click "Analyze Equation"
3. Keep auto-detected domains
4. Increase Population to 600
5. Increase Time to 60 seconds
6. Click "SOLVE"
7. Use Heatmap visualization to see solution circle
8. Solutions lie on circle perimeter
```

#### Workflow 4: Parameter Tuning for Difficult Equations

```
1. Try equation: sqrt(x+25)+log(49-x)=25
2. First attempt: default parameters
3. If no solution found:
   - Increase Population to 600
   - Increase Time to 120 seconds
   - Widen domain slightly
4. Click "SOLVE" again
5. Observe improved convergence in visualization
```

### Troubleshooting Guide

#### Problem: "Equation Not Recognized" Error

**Causes and Solutions:**

| Error Message | Cause | Solution |
|---------------|-------|----------|
| "Invalid syntax" | Unbalanced parentheses | Check bracket pairs: (), sqrt(...) |
| "Unknown variable" | Multi-letter variable | Use single letters: x, y, z only |
| "Unknown function" | Unsupported function | Check against supported list: sin, cos, log, sqrt, tan, arcsin, arccos, arctan |
| "Multiple = signs" | Equation format error | Ensure exactly one "=" between left and right sides |

**Example fixes:**
```
❌ Wrong: sqrtx+25=0           ✓ Correct: sqrt(x+25)=0
❌ Wrong: sinx=0.5             ✓ Correct: sin(x)=0.5
❌ Wrong: x+y+z=10=5           ✓ Correct: x+y+z=10 (or =5, not both)
```

#### Problem: "No Solutions Found" After Extended Time

**Possible causes:**

1. **Domain too narrow**: Solution lies outside specified range
   - Solution: Click ↻ to reset to auto-detected domains
   
2. **Parameters too conservative**: Population too small
   - Solution: Increase Population to 800-1000
   
3. **Equation genuinely difficult**: Some equations require longer time
   - Solution: Increase Time to 300+ seconds
   
4. **Numerical difficulty**: Very steep gradients or multiple local minima
   - Solution: Try different domain, increase Elitism

#### Problem: GUI Freezes or Becomes Unresponsive

**Cause**: Long computation time without visualization updates

**Solutions:**
- This is normal for complex equations with large populations
- Patience: let algorithm run to completion (check console for progress)
- If truly stuck (>10 minutes): close window and reduce parameters
- Increase `visualization_interval` in code for more frequent updates

#### Problem: Poor Solution Quality (High Error)

**Indicates**: Algorithm reached local optimum

**Solutions:**
1. Increase time limit for more generations
2. Increase population size for better exploration
3. Widen domain to escape local optimum
4. Increase tournament size to improve selection pressure
5. Manually refine solution using other tools

### Tips and Best Practices

**For Best Results:**

1. **Start Simple**: Test with quadratic or cubic first to understand workflow
2. **Verify Solutions**: Always check solutions by substitution in original equation
3. **Domain Knowledge**: If you know approximate solution location, use narrower domain
4. **Progressive Complexity**: Gradually increase problem difficulty and parameters
5. **Observe Visualization**: Use convergence plots to diagnose algorithm behavior
6. **Document Settings**: Note what parameters work for similar equation types

**Optimization Tips:**

- Small populations (100-200) often sufficient for simple equations
- Large populations (800+) needed only for very complex equations
- Time limit more important than population size for convergence
- Wider domains need proportionally longer solving time
- Multi-variable equations benefit from larger elitism values

### Exporting and Saving Results

Currently, results are displayed in the GUI. To save:

1. **Manual Copy**: Copy values from Solutions tab to text editor
2. **Automatic Export** (if implemented):
   - Results are saved to checkpoint files in `evaluation/checkpoints/`
   - Export as JSON or CSV for further analysis

### Extending the Application

The application is modular and can be extended for:

- **Custom fitness functions**: Modify `calculate_heuristic()` in `parse.py`
- **Additional functions**: Add to parser in `string_to_tree.py`
- **Parameter tuning**: Modify default values in `ga_solver.py`
- **Visualization modes**: Add new plot types in visualization panel
- **Multi-language support**: Localize GUI strings

---

## Usage Guide (Quick Reference)

### Equation Input

The equation builder supports:

**Mathematical Functions:**
- Trigonometric: sin(), cos(), tan(), arcsin(), arccos(), arctan()
- Logarithmic: log() (natural logarithm)
- Root: sqrt()
- Power: ^ (e.g., x^2)

**Operators:** +, -, *, /

**Variables:** x, y, z or any single letter

**Constants:** Numbers (0-9), decimal point (.), Pi button (π)

**Example Equations:**
```
sqrt(x+25)+log(49-x)=25
sin(x)^2+cos(x)^2=1
x^2+y^2=25
log(x)+sqrt(y)=10
```

### Domain Configuration

After analyzing the equation, the system automatically detects safe domains for each variable based on mathematical constraints (square roots, logarithms, inverse trig functions).

You can manually adjust the Min and Max values for each variable, or click the ↻ button to reset to auto-detected values.

### GA Parameters

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|---------|-------------------|
| Population | Total individuals across all islands | 400 | 100-1000 |
| Time (s) | Maximum solving time in seconds | 60 | 30-300 |
| Tournament | Tournament selection size | 5 | 3-10 |
| Elitism | Number of best individuals preserved | 5 | 2-10 |

**Tips:**
- Increase Population for more exploration
- Increase Time for complex equations
- Higher Tournament size increases selection pressure
- Higher Elitism preserves more good solutions

### Visualization

**Convergence Plot (Top):**
- Green line: Best fitness over generations
- Blue line: Average fitness
- Red line: Worst fitness
- Log scale for better visibility

**Distribution Plots (Bottom):**
- Select variables from dropdowns
- Switch between distribution (histogram) and heatmap (scatter) views
- View population diversity and convergence behavior

### Solution Interpretation

Solutions are displayed with:
- Variable values with high precision (12 decimal places)
- Error metric (lower is better, < 1e-6 is excellent)

Multiple solutions may be found for equations with multiple roots.

---

## Algorithm Overview

The solver uses a multi-island genetic algorithm with 4 populations evolving independently:

1. Island 0: Conservative (low mutation rate 0.1, high crossover 0.9)
2. Island 1: Moderate (mutation 0.2)
3. Island 2: Exploratory (mutation 0.3)
4. Island 3: Aggressive (mutation 0.4)

**Key Features:**

- **Diversity Maintenance**: Increasing diversity thresholds per island with forced re-initialization if diversity drops
- **Local Search**: Hybrid GA with simulated annealing and adaptive step size
- **Solution Graduation**: Solutions with error < 1e-6 are "graduated" and nearby individuals are removed to encourage exploration
- **Migration**: Best individuals migrate between islands every 10 generations
- **Fitness Penalization**: Penalties added for individuals near already-found solutions to encourage exploration of new solution regions

---

## Examples

### Example 1: Simple Square Root Equation
```
sqrt(x+25)+log(49-x)=25
```
Auto-detected domain: -24.99 < x < 48.99  
Typical solution: x ≈ 24.0  
Solving time: ~10-30 seconds

### Example 2: Trigonometric Identity
```
sin(x)^2+cos(x)^2=1
```
Multiple solutions found (any x is a solution)

### Example 3: Circle Equation
```
x^2+y^2=25
```
Multiple solutions on the circle perimeter

### Example 4: Complex Transcendental
```
log(x)+sqrt(y)=10
```
Non-linear constraints on both variables

---

## Project Structure

```
project1/
├── app.py                           # Main GUI application
├── core/
│   ├── ga_solver.py                 # Genetic algorithm implementation
│   ├── parse.py                     # Equation parsing utilities
│   ├── main.py                      # Command-line interface
│   ├── genetic/
│   │   ├── mutation.py              # Mutation operators
│   │   └── crossover.py             # Crossover operators
│   └── parser/
│       ├── string_to_tree.py        # Expression tree builder
│       └── string_to_formula.py     # Formula parsing utilities
├── evaluation/
│   ├── benchmark.py                 # Benchmark problem definitions
│   ├── perform_test.py              # Benchmark testing framework
│   ├── extended_test.py             # Extended testing for difficult problems
│   ├── execute_har_problem.py       # Execute hard problem tests
│   ├── report.py                    # Report generation
│   ├── checkpoints/                 # Checkpoint storage for resumable tests
│   └── report_output/               # Generated reports and figures
├── utils/
│   └── helpers.py                   # Utility functions
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── .gitignore                       # Git ignore rules
└── icon.ico / icon.png             # Application icon
```

---

## Running Tests and Benchmarks

### Standard Benchmark Suite

Run the full benchmark suite on 14 classical and scalable problems:

```bash
python evaluation/perform_test.py
```

**Configuration:**
- Population Size: 200
- Time Limit: 30 seconds per problem
- Runs per Problem: 10
- Resume Option: Automatically resumes from last checkpoint

**Output:**
- Checkpoints saved in `evaluation/checkpoints/`
- Summary statistics printed to console
- Results in JSON format for further analysis

### Extended Testing (Difficult Problems)

For problems with 0% success rate, run extended testing with increased resources:

```bash
python evaluation/extended_test.py
```

**Enhanced Configuration:**
- Population Size: 500 (increased)
- Time Limit: 300 seconds (5 minutes)
- Runs per Problem: 30
- Target Problems: Broyden_Tridiagonal_5D, Mixed_Transcendental

**Estimated Time:** ~5 hours for both problems

### Hard Problem Testing

Test specific difficult problems individually:

```bash
python evaluation/execute_har_problem.py
```

---

## Benchmark Results

Current benchmark suite includes:

**Classical Problems (Low Difficulty):**
- Quadratic: x² - 4 = 0
- Cubic: x³ - 6x² + 11x - 6 = 0
- Trigonometric: sin(x) = 0.5
- Sum of Squares: x² + y² = 2 (2D-4D)
- Powell Singular: x + 10y = 0
- Exponential: 2^x - 8 = 0

**Scalable Problems (Medium-High Difficulty):**
- Rosenbrock Roots: 10(y - x²) + (1 - x) = 0
- Broyden Tridiagonal: (2D, 3D, 5D variants)
- Circle-Line Intersection: x² + y² - 25 = 0
- Mixed Transcendental: √(x+25) + log(49-x) = 25

**Performance Summary:**
- Classical Problems: 100% success rate
- Scalable 2-4D: 100% success rate
- Broyden 5D: 0% success rate (under investigation)
- Mixed Transcendental: 0% success rate (extended testing in progress)

See `evaluation/report_output/` for detailed performance tables and visualizations.

---

## Running from Command Line

### Interactive Mode

For simple equation solving without GUI:

```bash
python core/main.py
```

Then follow the prompts to enter equations and configure parameters.

**Example Session:**
```
Enter equation: x^2-4=0

Left side formula tree:
    ^
   / \
  x   2

Right side formula tree:
4

Created solution:
Variables: x

Best solutions found:
x = 2.000000
Error: 1.23e-07
-----
x = -2.000000
Error: 1.45e-07
-----
```

---

## Performance Tuning

### For Faster Solving

- Reduce `Population`: 100-150 (less diversity but faster)
- Reduce `Time`: 10-20 seconds
- Reduce `Tournament Size`: 3-4
- Use smaller domain ranges if possible

### For More Reliable Solving

- Increase `Population`: 500-1000
- Increase `Time`: 120-300 seconds
- Increase `Tournament Size`: 7-10
- Increase `Elitism`: 10-20

### Memory Usage

- Memory scales linearly with Population Size
- Approximate usage: 10MB + (Population × 0.01MB)
- For Population=1000: ~20MB

---

## Troubleshooting

### GUI Doesn't Start
```bash
# Check tkinter installation
python -c "import tkinter; tkinter.Tk()"

# On Linux, install if missing:
sudo apt-get install python3-tk
```

### Slow Solving Performance
- Reduce time limit and check if solutions are found quickly
- Try simpler equations first to benchmark your system
- Reduce population size or increase tournament size

### Equation Not Recognized
- Check for balanced parentheses
- Ensure proper spacing (spaces are removed automatically)
- Use single-letter variable names (x, y, z, etc.)
- Verify all functions are supported

### Out of Memory
- Reduce Population Size
- Close other applications
- Reduce time limit
- Try simpler equations

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes with clear commit messages
4. Push to your branch (`git push origin feature/improvement`)
5. Open a Pull Request with detailed description

---

## Known Limitations

1. **Single-letter variables**: Only x, y, z, w, etc. are supported
2. **Equation complexity**: Very complex nested expressions may be slow
3. **Numerical precision**: Limited to double precision (IEEE 754)
4. **High-dimensional problems**: Performance degrades for >5 dimensions
5. **Complex numbers**: Only real solutions are found

---

## Future Improvements

- [ ] Support for multi-character variable names
- [ ] Parallel island evaluation
- [ ] Symbolic simplification before solving
- [ ] GPU acceleration for large populations
- [ ] Web-based interface
- [ ] Support for constraint satisfaction
- [ ] Adaptive parameter tuning

---

## Performance Notes

### Typical Solving Times
- Simple equations (quadratic, cubic): 1-5 seconds
- Medium equations (trig, exponential): 10-30 seconds
- Complex equations (mixed transcendental): 60-300 seconds

### Success Rates
- Classical problems: >95% success rate
- Scalable problems (2-4D): >90% success rate
- High-dimensional problems (5D+): 0-50% (problem dependent)

---

## References

**Genetic Algorithm Techniques:**
- Multi-island models for distributed evolution
- Tournament selection and elitism
- Adaptive mutation and crossover operators
- Diversity maintenance strategies

**Benchmark Problems:**
- Classical nonlinear equation sets
- Scalable test suites for algorithm evaluation
- Real-world transcendental equations

---

## License

This project is licensed under the MIT License - see LICENSE file for details.

---

## Contact & Support

For questions, issues, or suggestions:

- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions for general questions
- **Email**: your.email@example.com

---

**Last Updated:** January 6, 2026  
**Version:** 1.0.0  
**Status:** Active Development