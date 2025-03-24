import os
from fpdf import FPDF
from transformers import pipeline
import torch
import pandas as pd
from datetime import datetime

torch.manual_seed(42)

gpt2_pipe = pipeline("text-generation", model="gpt2")
CORRECTION_LOG_PATH = "llm_lib/reports/correction_log.csv"

def get_correction(deformation, model_choice="gpt2", for_pdf=False):
    prompt = f"A deformation of {deformation:.4f} mm was detected in a satellite. Suggest a professional correction plan."
    
    if model_choice == "gpt2":
        result = gpt2_pipe(prompt, max_length=100, do_sample=True)[0]['generated_text']
        return result
    else:
        return "⚠️ OpenAI quota exceeded. Switch to GPT-2 or check billing."


def export_correction_to_pdf(deformation, correction_text, image_path=None, include_image=True, diffusion_prompt=None):
    os.makedirs("llm_lib/reports", exist_ok=True)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="SmartSat-GenAI Correction Report", ln=True, align='C')
    pdf.ln(10)
    pdf.multi_cell(0, 10, f"Detected Deformation: {deformation:.4f} mm")
    pdf.ln(5)
    pdf.multi_cell(0, 10, "LLM Suggested Correction:")
    pdf.multi_cell(0, 10, correction_text)
    
    # Optionally add image from diffusion model
    if image_path and os.path.exists(image_path):
        pdf.ln(10)
        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(200, 10, txt="Diffusion-Generated Design", ln=True)
        pdf.image(image_path, x=10, w=180)
    
    # Include history summary
    pdf.ln(10)
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(200, 10, txt="Previous Corrections Log", ln=True)
    pdf.set_font("Arial", size=10)
    history_df = load_correction_history()
    if not history_df.empty:
        recent = history_df.tail(5)
        for _, row in recent.iterrows():
            pdf.multi_cell(0, 8, f"[{row['timestamp']}] Deformation: {row['deformation']} → {row['correction'][:80]}...")
    else:
        pdf.multi_cell(0, 10, "No previous corrections available.")

    # Optionally include the prompt
    if diffusion_prompt:
        pdf.ln(5)
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 10, f"Prompt used for generation: {diffusion_prompt}")
    
    path = "llm_lib/reports/correction_report.pdf"
    pdf.output(path)
    return path

def log_correction(deformation, correction_text):
    os.makedirs("llm_lib/reports", exist_ok=True)
    log_entry = pd.DataFrame({
            "timestamp": [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            "deformation": [deformation],
            "correction": [correction_text]
        })

    if os.path.exists(CORRECTION_LOG_PATH):
        log_entry.to_csv(CORRECTION_LOG_PATH, mode='a', index=False, header=False)
    else:
        log_entry.to_csv(CORRECTION_LOG_PATH, index=False, header=True)

def load_correction_history():
    if os.path.exists(CORRECTION_LOG_PATH):
        return pd.read_csv(CORRECTION_LOG_PATH)
    else:
        return pd.DataFrame(columns=["timestamp", "deformation", "correction", "prompt"])
    
if __name__ == "__main__":
    sample_deform = 0.0456
    correction_ui = get_correction(sample_deform, for_pdf=False)
    correction_pdf = get_correction(sample_deform, for_pdf=True)
    diffusion_prompt = "3D printed modular satellite component, max temp 90°C, lightweight, efficient heat dissipation"
    image_path = "generative_design/outputs/diffusion_sat_component.png"
    export_path = export_correction_to_pdf(sample_deform, correction_pdf, image_path=image_path, diffusion_prompt=diffusion_prompt)
    log_correction(sample_deform, correction_pdf, diffusion_prompt=diffusion_prompt)
    print(f"PDF report saved to {export_path}")
    print("Correction for UI:", correction_ui)
    print("Correction logged.")