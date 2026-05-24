# Import necessary modules and classes
from generate_data import generate_data
from problem import OptimizationProblem
from simannealing import SimulatedAnnealing

# Set parameters
seed = "1003010"             # Seed ID
n = 100                      # Problem size (100x100 grid)
initial_temp = 100           # Initial temperature
cooling_rate = 0.995         # Cooling rate
max_steps = 5000             # Number of steps

# Generate the cost matrix C
C = generate_data(n, seed)

# Initialize the optimization problem with the cost matrix
problem = OptimizationProblem(C)

# Initialize the simulated annealing algorithm
sa = SimulatedAnnealing(problem, initial_temp, cooling_rate, max_steps)

# Run the algorithm
best_x, best_y, best_val = sa.run()

# Print final results
print(f"\nBest result found: f({best_x}, {best_y}) = {best_val:.4f}")
print(f"Actual minimum value in matrix: {C.min():.4f}")
