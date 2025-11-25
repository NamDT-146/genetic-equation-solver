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
pip install numpy matplotlib scipy
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

## Usage Guide

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

- Diversity Maintenance: Increasing diversity thresholds per island with forced re-initialization if diversity drops
- Local Search: Hybrid GA with simulated annealing and adaptive step size
- Solution Graduation: Solutions with error < 1e-6 are "graduated" and nearby individuals are removed
- Migration: Best individuals migrate between islands every 10 generations
- Fitness Penalization: Penalties added for individuals near already-found solutions to encourage exploration

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
├── app.py                      # Main GUI application
├── core/
│   ├── ga_solver.py            # Genetic algorithm implementation
│   ├── parse.py                # Equation parsing utilities
│   ├── solution.py             # Solution class definition
│   ├── genetic/
│   │   ├── mutation.py         # Mutation operators
│   │   └── crossover.py        # Crossover operators
│   └── parser/
│       └── string_to_tree.py   # Expression tree builder
├── README.md
├── .gitignore
└── icon.ico / icon.png
```

---

## License

This project is licensed under the MIT License.

---

## Contact

For questions, issues, or suggestions, please open an issue on GitHub.