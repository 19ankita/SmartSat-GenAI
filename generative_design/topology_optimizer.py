import os
import numpy as np
import matplotlib.pyplot as plt
import trimesh

def generate_structure(length=1.0, resolution=100, max_temp=80, weight_constraint=0.5):
    """
    Simulates an optimized structure under thermal and weight constraints.
    Returns coordinates representing the optimized profile.
    """
    x = np.linspace(0, length, resolution)
    
    # Simulate thermal influence and structural optimization
    thermal_profile = np.exp(-3 * x) * np.sin(5 * np.pi * x)
    weight_factor = (1 - weight_constraint)  # lower weight = more aggressive shape

    # Deformation-inspired optimized shape
    y = weight_factor * thermal_profile + np.random.normal(0, 0.01, size=x.shape)
    return x, y

def visualize_structure(x, y, show=True):
    plt.plot(x, y, label="Generated Profile")
    plt.xlabel("Length (normalized)")
    plt.ylabel("Stress Response / Shape")
    plt.title("Simulated Generative Structure")
    plt.grid(True)
    plt.legend()
    if show:
        plt.show()

def export_to_mesh(x, y, filename="generative_structure.obj"):
    """
    Converts the 2D structure into a simple 3D mesh using trimesh.
    """
    z = np.zeros_like(x)
    vertices = np.stack([x, y, z], axis=1)
    
    # Create lines connecting each point
    faces = [[i, i+1, i+2] for i in range(len(vertices)-2)]
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    os.makedirs("generative_design/exports", exist_ok=True)
    mesh.export(f"generative_design/exports/{filename}")
    print(f"Mesh exported to generative_design/exports/{filename}")

if __name__ == "__main__":
    x, y = generate_structure(length=1.0, resolution=100, max_temp=80, weight_constraint=0.4)
    visualize_structure(x, y)
    export_to_mesh(x, y)
