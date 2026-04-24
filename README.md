# House-price-prediction
End-to-end machine learning pipeline for house price prediction with feature engineering, model comparison, and explainability using SHAP

This project predicts house prices using machine learning models trained on the Ames Housing Dataset.

It demonstrates an end-to-end ML pipeline including:
- Data preprocessing
- Feature engineering
- Model training
- Evaluation
- Deployment using Streamlit

---

##  Dataset
- Ames Housing Dataset
- Contains 80+ features describing residential homes

---

##  Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Joblib

---

##  Model
- Random Forest Regressor
- Handles non-linear relationships and large feature sets

---

##  Performance
- RMSE: ~26,800


---

##  Features
- Handles missing values
- Encodes categorical variables using one-hot encoding
- Uses feature importance for interpretability

---

##  Web App
A simple Streamlit app allows users to input house features and get price predictions.

### Run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
