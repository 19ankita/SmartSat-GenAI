import os
import glob
import joblib
import json
from datetime import datetime

TRACKER_PATH = "ml_model/trained_models/latest_model.txt"

def reset_model_tracker():
    """Deletes the tracker file before new training begins (once per session)."""
    if os.path.exists(TRACKER_PATH):
        os.remove(TRACKER_PATH)

def append_to_model_tracker(model_path):
    """Appends a new model path to the tracker file."""
    with open(TRACKER_PATH, "a") as f:
        f.write(model_path + "\n")

def clean_model_directory(directory="ml_model/trained_models", keep_latest_n=0, model_prefix=None):
    os.makedirs(directory, exist_ok=True)
    
    # Filter only models with the given prefix (e.g., LR_model or RF_model)
    models = sorted(
        glob.glob(os.path.join(directory, f"{model_prefix}_*.pkl")),
        reverse=True
    )
    
    for model_path in models[keep_latest_n:]:
        os.remove(model_path)
        meta_path = model_path.replace(".pkl", "_meta.json")
        if os.path.exists(meta_path):
            os.remove(meta_path)
            
    # Also remove any stray .json metadata files that aren't linked to existing models
    all_meta_files = glob.glob(os.path.join(directory, f"{model_prefix}_*_meta.json"))
    for meta_file in all_meta_files:
        pkl_path = meta_file.replace("_meta.json", ".pkl")
        if not os.path.exists(pkl_path):
            os.remove(meta_file)     

def save_model_and_metadata(model, model_name, r2_score, features, directory="ml_model/trained_models"):
    os.makedirs(directory, exist_ok=True)
    
    # Clean old models first
    clean_model_directory(directory=directory, keep_latest_n=0, model_prefix=model_name)  # keep the latest before saving a new one
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{model_name}_{timestamp}.pkl"
    model_path = os.path.join(directory, model_filename)

    # Save the model
    joblib.dump(model, model_path)

    # Save the metadata
    metadata = {
        "timestamp": timestamp,
        "model_file": model_filename,
        "r2_score": r2_score,
        "features": features,
        "target": "deformation"
    }
    meta_path = model_path.replace(".pkl", "_meta.json")
    with open(meta_path, "w") as meta_file:
        json.dump(metadata, meta_file, indent=4)

    return model_path
