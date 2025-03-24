# generative_design/topology_optimizer.py
import numpy as np
import matplotlib.pyplot as plt

def generate_structure(seed=0):
    np.random.seed(seed)
    x = np.linspace(0, 1, 100)
    y = np.sin(4 * np.pi * x) * np.exp(-3 * x) + np.random.normal(0, 0.02, 100)
    return x, y

if __name__ == "__main__":
    x, y = generate_structure()
    plt.plot(x, y)
    plt.title("Simulated Optimized Structure")
    plt.xlabel("Normalized Length")
    plt.ylabel("Stress Response")
    plt.grid(True)
    plt.show()

