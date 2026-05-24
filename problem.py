import numpy as np

# Defines the optimization problem
class OptimizationProblem:
    def __init__(self, C):
        self.C = C # Cost matrix
        self.n = C.shape[0]  # Grid size

    # Returns the cost at position (x, y) using modulo to handle border wrapping
    def evaluate(self, x, y):
        return self.C[x % self.n, y % self.n]

    # Returns a random neighboring coordinate with modulo-n wrapping
    def get_random_neighbor(self, x, y):
        xp = np.random.choice([(x - 1) % self.n, (x + 1) % self.n])
        yp = np.random.choice([(y - 1) % self.n, (y + 1) % self.n])
        return xp, yp
