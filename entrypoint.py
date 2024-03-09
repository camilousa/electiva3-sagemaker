import joblib
import os

def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

def predict_fn(input_data, model):
     if len(input_data.shape) == 1:
      proba = model.predict_proba(input_data.reshape(-1, 1).T)
     else:
      proba = model.predict_proba(input_data)   
     return proba
