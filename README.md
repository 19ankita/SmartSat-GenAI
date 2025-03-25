🚀 SmartSat-GenAI
## 🌌 About the Project

**SmartSat-GenAI** is a powerful, AI-driven simulation and visualization platform for **smart satellite design, deformation analysis, and correction planning**, integrating:

- 🧠 **Large Language Models (LLMs)** for intelligent correction suggestions  
- 🎨 **Diffusion Models** for generative design visualization  
- 📈 **Sensor-driven ML models** for predicting structural deformation  

This project is part of a broader initiative to combine **Generative AI with Aerospace Engineering**, inspired by the [**ESAF Project**](https://www.reflexaerospace.com/additive-manufacturing) and Reflex Aerospace’s novel approach to achieving structural precision through **additive manufacturing** and **embedded sensor feedback**.

> 🚀 Designed for real-world satellite scenarios — custom, modular, efficient, and built for the future of space.
---

## 🧠 Project Overview

SmartSat-GenAI bridges aerospace and AI using:

  - Sensor-based ML models to predict deformation

  - LLMs (GPT-2 / OpenAI) to suggest engineering corrections

  - Diffusion Models for 3D printable component design

  - Generative geometry to simulate and export optimized structures

---

## ✨ Key Features

- 📈 **Sensor Data Analysis** – Analyze thermal sensor data and predict structural deformation.
- 🤖 **Model Comparison** – Train and evaluate ML models (Linear Regression, Random Forest) on satellite datasets.
- 📊 **Residual & Variability Plots** – Visual diagnostics and metrics like R², variance, and std dev.
- 🧠 **LLM Correction Generator** – GPT-2 (offline) and OpenAI GPT-3.5 (online) support for professional correction suggestions.
- 📄 **PDF Report Export** – Auto-generates a printable PDF report with the timestamp, correction, and historical summary.
- 🖼️ **Generative Structure Design** – Interactive mesh generation via sliders (PyVista/Plotly).
- 🎨 **Diffusion-Based Concept Generator** – Generate 3D printable designs with prompts like _“lightweight heat-optimized star tracker frame.”_

---

## Project Workflow

[SENSOR DATA CSV]
        │
        ▼
[ML MODEL TRAINING]
 (Linear Regression / Random Forest)
        │
        ▼
[PREDICTED DEFORMATION]
        │
        ▼
[LLM CORRECTION SUGGESTION]
 (GPT-2 / OpenAI GPT-3.5)
        │
        ├────► [DIFFUSION PROMPT GENERATION]
        │            │
        │            ▼
        │     [3D DESIGN IMAGE 🖼️]
        │
        ▼
[GENERATIVE GEOMETRY (STRUCTURE DESIGN)]
        │
        ▼
[PDF REPORT 🧾 + CORRECTION LOG 📜]



---               

## 📂 Project Structure

```bash
.
├── data/                      # Sensor CSV datasets
├── dashboard/                 # Streamlit UI
├── ml_model/                  # ML models, training scripts, utils, correction model
├── generative_design/
│   ├── topology_optimizer.py  # Simulates optimized satellite structure
│   └── diffusion_model/
│       └── diffusion_generator.py
├── scripts/                   # End-to-end pipelines & CLI utilities
├── .env                       # API keys (excluded from repo)
├── requirements.txt
└── README.md

---


## ⚙️ Installation

# Clone the repo
git clone https://github.com/19ankita/SmartSat-GenAI.git
cd SmartSat-GenAI

# Create virtual environment
conda create -n smartsat python=3.8
conda activate smartsat

# Install dependencies
pip install -r requirements.txt


---


## 🚀 Running the App
# Start the Streamlit dashboard
streamlit run dashboard/streamlit_app.py

## Note : If you do not have OpenAI access, the app will automatically fall back to GPT-2.


---


## 📊 Model Training
Train models independently:
python ml_model/train_linear_model.py
python ml_model/train_rf_model.py


---


## 🛠️ Tools & Libraries

| Category               | Libraries / Tools                                             | Purpose                                      |
|------------------------|---------------------------------------------------------------|----------------------------------------------|
| **UI & Visualization** | `streamlit`, `matplotlib`, `seaborn`, `plotly`               | Interactive dashboard and data plotting      |
| **ML & Modeling**      | `scikit-learn`, `joblib`, `pandas`                           | Model training, persistence, and processing  |
| **Generative AI**      | `transformers`, `diffusers`, `torch`                         | LLM-based corrections & diffusion generation |
| **Reporting**          | `fpdf`                                                       | PDF correction report generation             |
| **Secrets Handling**   | `python-dotenv` (`dotenv`)                                   | Secure environment variable management       |

---


## 📄 Sample Output

- 🧾 **Correction Report (PDF)**  
  Automatically generated report with timestamp, LLM suggestion, and correction history summary.

- 🖼️ **Diffusion-Generated Image**  
  3D printable design concept based on prompt generated by LLM or user input.

- 🌐 **Real-time 3D Preview**  
  Mesh structure visualization via Plotly for satellite components.

- 🗂️ **Logged Correction History (CSV)**  
  Timestamped log of deformation values and their suggested corrections.

---

## 🧪 Future Roadmap

- ✅ Add timestamped correction logging  
- [ ] 🔁 **Add HuggingFace model selection UI**  
- [ ] 🌡️ **Integrate thermal simulation physics backend**  
- [ ] 🛰️ **Extend to satellite subsystem design**  
  - e.g., antennas, thrusters, modular payloads  
- [ ] 🚀 **Deploy on HuggingFace Spaces for live demo**


