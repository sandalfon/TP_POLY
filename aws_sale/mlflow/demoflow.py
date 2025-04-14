import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
        getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("my_first_experiment")

data = fetch_california_housing()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

params = {
    "n_estimators": 100,
    "max_depth": 6,
    "random_state": 42
}

with mlflow.start_run():
    mlflow.log_params(params)


    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)


    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    mlflow.log_metric("mse", mse)

    mlflow.sklearn.log_model(model, "model")