import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import matplotlib.pyplot as plt
from generative_design.topology_optimizer import generate_structure, visualize_structure

st.set_page_config(page_title="Generative Structure Preview", layout="centered")
st.title("Generative Design Simulator")

st.markdown("""
Adjust the sliders on the left to simulate a generative satellite component structure 
under different thermal and weight constraints.
""")

# Sidebar parameters
st.sidebar.header("Design Constraints")

length = st.sidebar.slider("Structure Length", 0.1, 5.0, value=1.0, step=0.1)
resolution = st.sidebar.slider("Mesh Resolution", 10, 300, value=100, step=10)
max_temp = st.sidebar.slider("Max Operating Temperature (°C)", 10, 150, value=80)
weight_constraint = st.sidebar.slider("Weight Constraint", 0.1, 1.0, value=0.5, step=0.05)

# Generate structure
x, y = generate_structure(
    length=length,
    resolution=resolution,
    max_temp=max_temp,
    weight_constraint=weight_constraint
)

# Display chart
st.subheader("Optimized Structure Profile")
fig, ax = plt.subplots()
ax.plot(x, y, color="royalblue", linewidth=2)
ax.set_xlabel("Length (normalized)")
ax.set_ylabel("Stress Response / Shape")
ax.set_title("Simulated Generative Structure")
ax.grid(True)
st.pyplot(fig)

# Export button (optional future step)
st.markdown("---")
if st.button("Export as OBJ"):
    from generative_design.topology_optimizer import export_to_mesh
    export_to_mesh(x, y)
    st.success("Mesh exported to generative_design/exports/generative_structure.obj")
