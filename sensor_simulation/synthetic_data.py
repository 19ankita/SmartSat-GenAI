import numpy as np
import pandas as pd
import os

# Create the directory if it doesn't exist
os.makedirs("data/synthetic", exist_ok=True)

def generate_sensor_data(samples=1000):
    # Simulate temperature (T) and deformation (ΔL)
    T = np.random.uniform(-50, 100, samples)  # Temperature range (°C)
    alpha = 23e-6  # Thermal expansion coefficient (aluminum)
    L = 1.0        # Initial length (m)
    delta_L = alpha * L * T + np.random.normal(0, 0.0001, samples)  # Add noise
    df = pd.DataFrame({"temperature": T, "deformation": delta_L})
    return df

if __name__ == "__main__":
    df = generate_sensor_data()
    df.to_csv("data/synthetic/sample_data.csv", index=False)
    print("Sensor data saved to data/synthetic/sample_data.csv")