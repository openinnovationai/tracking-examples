import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from oip_tracking_client.tracking import TrackingClient

api_host = "http://<oi_platform>/api"

api_key = "<api_key>"

workspace_name = "<workspace_name>"

experiment_name = "<experiment_name>"

# set up TrackingClient
TrackingClient.connect(api_host, api_key, workspace_name)

# set the experiment
TrackingClient.set_experiment(experiment_name)


# Load the dataset and split it into training and testing sets.
db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

with TrackingClient.start_run():

    # Set the run name
    TrackingClient.set_run_name("Run XGDBoost 1")

    # Create and train the XGBoost model.
    params = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "objective": "reg:squarederror",
    }
    xgb_model = xgb.XGBRegressor(**params)
    xgb_model.fit(X_train, y_train)

    signature = TrackingClient.infer_signature(X_train, y_train)
    TrackingClient.xgboost.log_model(xgb_model, "model", signature=signature)
