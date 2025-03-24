import os
import numpy as np
import pandas as pd

# Create the directory if it doesn't exist
os.makedirs("data/synthetic", exist_ok=True)

def simulate_sensor_data(num_points=500):
    temperature = np.random.normal(35, 5, num_points)
    deformation = 0.001 * temperature + np.random.normal(0, 0.0005, num_points)
    return pd.DataFrame({"temperature": temperature, "deformation": deformation})

if __name__ == "__main__":
    df = simulate_sensor_data()
    df.to_csv("data/synthetic/sensor_data.csv", index=False)
    print("Sensor data saved to data/synthetic/sensor_data.csv")