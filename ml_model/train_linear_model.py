import os
import glob
import pandas as pd

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from ml_model.utils import append_to_model_tracker, save_model_and_metadata

def train_model(data_path):    
    df = pd.read_csv(data_path)
    X = df[["temperature"]]
    y = df["deformation"]
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    # Save model + metadata
    model_path = save_model_and_metadata(
        model=model,
        model_name="LR_model",
        r2_score=r2,
        features=list(X.columns)
    )

    print(f"Model saved to {model_path} with R²: {r2:.4f}")
    
    append_to_model_tracker(model_path) 
    return y_pred

if __name__ == "__main__":
    train_model("data/synthetic/sensor_data_LR.csv")