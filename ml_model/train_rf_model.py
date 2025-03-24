import os
import sys
import glob
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from ml_model.utils import append_to_model_tracker, save_model_and_metadata

def train_model(data_path):    
    df = pd.read_csv(data_path)
    X = df[["temperature"]]
    y = df["deformation"]
    
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    # Save model + metadata
    model_path = save_model_and_metadata(
        model=model,
        model_name="RF_model",
        r2_score=r2,
        features=list(X.columns)
    )

    print(f"Model saved to {model_path} with R²: {r2:.4f}")
    
    append_to_model_tracker(model_path)
    return y_pred

if __name__ == "__main__":
    train_model("data/synthetic/sensor_data_RF.csv")