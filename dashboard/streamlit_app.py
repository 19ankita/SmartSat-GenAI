import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.graph_objects as go
from ml_model.train_linear_model import train_model as train_linear_model
from ml_model.train_rf_model import train_model as train_rf_model
from sklearn.metrics import r2_score
from generative_design.topology_optimizer import generate_structure, export_to_mesh
from llm_lib.correction_model import get_correction, export_correction_to_pdf
from generative_design.diffusion_model.diffusion_generator import generate_satellite_design

st.set_page_config(page_title="SmartSat GenAI Dashboard", layout="centered")
st.title("\U0001F680 SmartSat-GenAI Dashboard")

# Tab navigation
tab1, tab2 = st.tabs(["Sensor Analysis + Corrections", "Generative Structure Preview"])

with tab1:
    st.header("📊 Sensor-Based Deformation and Correction")
    st.markdown("""Upload a CSV file containing sensor data to predict deformation and suggest corrections.""")
    
    # Multiple dataset support
    dataset_files = sorted([f for f in os.listdir("data/synthetic") if f.endswith(".csv")])
    selected_dataset = st.selectbox("Select Sensor Dataset:", dataset_files)
    dataset_path = os.path.join("data/synthetic", selected_dataset)
    df = pd.read_csv(dataset_path)
    st.write("## Sensor Data", df.head())
      
    st.markdown("---")
    st.subheader("📉 Is deformation linearly dependent on temperature?")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(data=df, x="temperature", y="deformation", ax=ax1)
    ax1.set_title("Temperature vs Deformation")
    st.pyplot(fig1)
    
    st.subheader("📦 Outlier Detection & Data Summary")
    st.write(df.describe())

    st.subheader("📊 Data Variability")
    st.write("**Standard Deviation:**")
    st.write(df.std())
    st.write("**Variance:**")
    st.write(df.var())
    
    st.markdown("---")
    st.subheader("Compare Model Performance")

    X = df[["temperature"]]
    y = df["deformation"]

    y_pred_lr = train_linear_model(dataset_path)
    df["predicted_deformation"] = y_pred_lr
    r2_lr = r2_score(y, y_pred_lr)

    y_pred_rf = train_rf_model(dataset_path)
    df["predicted_deformation"] = y_pred_rf
    r2_rf = r2_score(y, y_pred_rf)

    st.success(f"📈 Linear Regression R²: {r2_lr:.4f} | Random Forest R²: {r2_rf:.4f}")

    st.subheader("📉 Prediction Plot (Side-by-Side)")
    fig2, ax2 = plt.subplots()
    ax2.scatter(df["temperature"], df["deformation"], label="Actual", color="gray", alpha=0.5)
    ax2.plot(df["temperature"], y_pred_lr, label="Linear Regression", color="blue", linestyle="--")
    ax2.plot(df["temperature"], y_pred_rf, label="Random Forest", color="green", linestyle="-", alpha=0.5)
    ax2.set_xlabel("Temperature")
    ax2.set_ylabel("Deformation")
    ax2.set_title("Model Predictions vs Actual")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.3)
    st.pyplot(fig2)

    st.subheader("🔍 Residual Plots")
    fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(10, 4))
    ax3.scatter(y_pred_lr, y - y_pred_lr, color="blue", alpha=0.5)
    ax3.axhline(0, color="black", linestyle="--")
    ax3.set_title("Linear Regression Residuals")
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Residuals")

    ax4.scatter(y_pred_rf, y - y_pred_rf, color="green", alpha=0.5)
    ax4.axhline(0, color="black", linestyle="--")
    ax4.set_title("Random Forest Residuals")
    ax4.set_xlabel("Predicted")
    ax4.set_ylabel("Residuals")
    st.pyplot(fig3)

    
    # '# Multiple model selection
    # model_files = sorted([f for f in os.listdir("ml_model/trained_models") if f.endswith(".pkl")], reverse=True)
    # selected_model_file = st.selectbox("Select Trained Model to Load:", model_files)
    # selected_model_path = os.path.join("ml_model/trained_models", selected_model_file)
    # model = joblib.load(selected_model_path)
    # model_choice = st.selectbox("Choose LLM for Correction:", ["openai", "gpt2"])

    # # Load metadata if available
    # meta_file_path = selected_model_path.replace(".pkl", "_meta.json")
    # if os.path.exists(meta_file_path):
    #     with open(meta_file_path, "r") as f:
    #         metadata = json.load(f)
    #     st.info(f"🔄 Using model: `{selected_model_file}` | R² Score: {metadata['r2_score']:.4f}")
    # else:
    #     st.info(f"🔄 Using model: `{selected_model_file}`")

    # df["predicted_deformation"] = model.predict(df[["temperature"]])
 
    # # Calculate R² score for current dataset + model
    # r2 = r2_score(df["deformation"], df["predicted_deformation"])
    # st.success(f"📈 R² Score for this run: {r2:.4f}")

    # # Visualization with R² in title
    # st.subheader("📉 Temperature vs Predicted Deformation")

    # fig, ax = plt.subplots(figsize=(8, 5))

    # # Actual values as scatter
    # ax.scatter(
    #     df["temperature"], df["deformation"],
    #     color='steelblue', s=10, label="Actual Deformation", alpha=0.8, edgecolors='black', marker='o'
    # )

    # # Predicted values as transparent line on top
    # ax.plot(
    #     df["temperature"], df["predicted_deformation"],
    #     color='darkred', linestyle='-', linewidth=0.5, label="Predicted Deformation", alpha=0.6
    # )

    # # Axis labels and title
    # ax.set_xlabel("Temperature (°C)", fontsize=12)
    # ax.set_ylabel("Deformation (mm)", fontsize=12)
    # ax.set_title(f"Model Prediction vs Actual (R² = {r2:.4f})", fontsize=14, fontweight='bold')

    # # Legend and grid
    # ax.legend()
    # ax.grid(True, linestyle='--', alpha=0.3)

    # st.pyplot(fig)

    # Show correction output for a sample value
    deformation_sample = df["predicted_deformation"].iloc[0]
    st.subheader("\U0001F52E LLM-Based Correction Suggestion")
    model_choice = st.selectbox("Choose LLM for Correction:", ["gpt2", "openai"])
    correction = get_correction(deformation_sample, model_choice)
    st.code(correction, language="text")
    st.markdown(f"### Deformation Correction: **{correction}**")
    
    if "correction" not in st.session_state:
        st.session_state.correction = get_correction(deformation_sample, model_choice)

    correction = st.session_state.correction

    
    if st.button("Export Correction Report to PDF"):
        path = export_correction_to_pdf(deformation_sample, correction)
        with open(path, "rb") as file:
            st.download_button(label="📄 Download PDF Report", data=file, file_name="correction_report.pdf")
            
    try:
        correction = get_correction(deformation_sample, model_choice)
    except openai.RateLimitError:
        st.error("⚠️ OpenAI API quota exceeded. Please check your usage and billing.")
        correction = "Quota exceeded"

    
        
with tab2:
    st.header("Generative Structure Design")
    st.markdown("""Adjust constraints below to simulate optimized geometry for satellite components.""")

    # Sliders for constraints
    length = st.slider("Structure Length", 0.1, 5.0, value=1.0, step=0.1)
    resolution = st.slider("Mesh Resolution", 10, 300, value=100, step=10)
    max_temp = st.slider("Max Operating Temperature (°C)", 10, 150, value=80)
    weight_constraint = st.slider("Weight Constraint", 0.1, 1.0, value=0.5, step=0.05)

    x, y = generate_structure(length, resolution, max_temp, weight_constraint)

    fig3d = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=[0]*len(x),
        mode='lines+markers',
        line=dict(color='royalblue', width=4),
        marker=dict(size=3)
    )])
    fig3d.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        ),
        width=700,
        margin=dict(r=20, b=10, l=10, t=10)
    )

    st.plotly_chart(fig3d)

    if st.button("Export STL"):
        export_to_mesh(x, y, filename="generative_structure.stl")
        st.success("STL exported to generative_design/exports/generative_structure.stl")
        
with st.sidebar.expander("Generate Structure Concept (Diffusion)"):
    diffusion_prompt = st.text_input("Base Description", value="3D printed satellite component optimized for thermal performance")
    max_temp = st.slider("Max Operating Temp (°C)", min_value=50, max_value=200, value=120, step=10)
    
    full_prompt = f"{diffusion_prompt}, max temp {max_temp}°C, modular design, optimized for heat dissipation"

    if st.button("Generate Concept Image"):
        st.write("Prompt sent to diffusion model:")
        st.code(full_prompt, language="text")

        image_path = generate_satellite_design(prompt=full_prompt, max_temp=max_temp)
        st.image(image_path, caption="Diffusion-Generated Design")
      
