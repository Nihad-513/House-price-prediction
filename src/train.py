import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import joblib
import os
import numpy as np
from .preprocessing import load_data, preprocess


#print("Current working directory:", os.getcwd())
#print("Data folder exists:", os.path.exists("data"))
#print("Raw folder exists:", os.path.exists("data/raw"))
#print("File exists:", os.path.exists("data/raw/train.csv"))
#print(os.listdir("data/raw"))
def train_model():

    df = load_data("data/raw/train.csv")

    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    X = preprocess(X)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )


    model.fit(X_train, y_train)



    preds = model.predict(X_test)

    from sklearn.metrics import mean_squared_error
    import numpy as np
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print("RMSE:", rmse)

    import joblib
    joblib.dump(model, "models/model.pkl")

    return model

if __name__ == "__main__":
    train_model()