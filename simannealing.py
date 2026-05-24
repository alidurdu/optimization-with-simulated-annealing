import numpy as np
import matplotlib.pyplot as plt

# Simulated Annealing algorithm
class SimulatedAnnealing:
    def __init__(self, problem, initial_temp=100, cooling_rate=0.99, max_steps=10000):
        self.problem = problem                  # The problem to optimize
        self.T = initial_temp                   # Starting temperature
        self.cooling_rate = cooling_rate        # Temperature reduction factor
        self.max_steps = max_steps              # Total number of iterations
        self.history = []                       # Stores function values over time
        self.acceptance_probs = []              # Stores acceptance probabilities

    # Runs the simulated annealing optimization loop
    def run(self, verbose=True, plot=True):
        x = np.random.randint(0, self.problem.n)  # Random initial x
        y = np.random.randint(0, self.problem.n)  # Random initial y
        current_value = self.problem.evaluate(x, y)

        for step in range(self.max_steps):
         # Propose a new neighbor using modulo-n wrapped grid            
            xp, yp = self.problem.get_random_neighbor(x, y)
            new_value = self.problem.evaluate(xp, yp)

            # Calculate acceptance probability
            delta = new_value - current_value
            acceptance_prob = min(1, np.exp(-delta / self.T))
 
            # Accept or reject the new point
            if np.random.rand() < acceptance_prob:
                x, y = xp, yp
                current_value = new_value

            self.history.append(current_value)
            self.acceptance_probs.append(acceptance_prob)

            # Reduce the temperature
            self.T *= self.cooling_rate

            if verbose and step % 1000 == 0:
                print(f"Step {step}: Value = {current_value:.4f}, Temp = {self.T:.4f}")

        if plot:
            self.plot_results()

        return x, y, current_value

    # Plots function value and acceptance probability over time
    def plot_results(self):
        plt.figure(figsize=(12, 5))

        # Plot function value
        plt.subplot(1, 2, 1)
        plt.plot(self.history)
        plt.title("Function Value Over Iterations")
        plt.xlabel("Step")
        plt.ylabel("f(x, y)")

        # Plot acceptance probabilities
        plt.subplot(1, 2, 2)
        plt.plot(self.acceptance_probs)
        plt.title("Acceptance Probability Over Iterations")
        plt.xlabel("Step")
        plt.ylabel("Acceptance Probability")

        plt.tight_layout()
        plt.show()
