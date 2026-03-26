import os
import joblib

def load_model_joblib(path) :
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    model = joblib.load(path)
    print(f"Model loaded from: {path}")
    return model