import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from ml_model.train_linear_model import train_model as train_lr
from ml_model.train_rf_model import train_model as train_rf
from llm_lib.correction_model import get_correction, export_correction_to_pdf, log_correction
from generative_design.diffusion_model.diffusion_generator import generate_satellite_design

from datetime import datetime
import os
import argparse

def basic_keywords_from_llm(llm_text):
    text = llm_text.lower()
    keywords = []
    if "thermal" in text:
        keywords.append("thermal optimization")
    if "mount" in text:
        keywords.append("modular mounting")
    if "lightweight" in text or "mass" in text:
        keywords.append("lightweight material")
    if not keywords:
        keywords.append("high performance")
    return ", ".join(keywords)

def run_pipeline(data_path, model_type="lr", model_choice="gpt2"):
    # === Step 1: Load Sensor Data ===
    df = pd.read_csv(data_path)
    X = df[["temperature"]]
    y = df["deformation"]

    # === Step 2: Train + Predict ===
    if model_type == "lr":
        y_pred = train_lr(data_path)
    elif model_type == "rf":
        y_pred = train_rf(data_path)
    else:
        raise ValueError("Unsupported model type.")

    df["predicted_deformation"] = y_pred
    deformation_sample = df["predicted_deformation"].iloc[0]

    # === Step 3: LLM Correction ===
    correction_text = get_correction(deformation_sample, model_choice=model_choice)

    # === Step 4: Generate Prompt from LLM Output ===
    prompt = f"3D printed satellite component with {basic_keywords_from_llm(correction_text)}"
    print("📌 Prompt for Diffusion Model:", prompt)

    # === Step 5: Run Diffusion Model ===
    image_path = generate_satellite_design(prompt)

    # === Step 6: Save PDF Report ===
    pdf_path = export_correction_to_pdf(deformation_sample, correction_text, include_image=True)

    # === Step 7: Log the Correction ===
    log_correction(deformation_sample, correction_text)

    print(f"✅ Pipeline completed. PDF: {pdf_path}")
    print(f"🖼️ Image: {image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the SmartSat-GenAI pipeline")
    parser.add_argument("--data", type=str, required=True, help="Path to the sensor dataset CSV")
    parser.add_argument("--model", type=str, choices=["lr", "rf"], default="lr", help="Model type to train (lr or rf)")
    parser.add_argument("--llm", type=str, choices=["gpt2", "openai"], default="gpt2", help="LLM for correction")

    args = parser.parse_args()
    run_pipeline(data_path=args.data, model_type=args.model, model_choice=args.llm)
