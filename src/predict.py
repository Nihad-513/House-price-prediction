import joblib
import pandas as pd

model = joblib.load("model.pkl")

def predict(input_df):
    return model.predict(input_df)