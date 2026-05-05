from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from oip_tracking_client.tracking import TrackingClient
from sklearn.metrics import mean_squared_error, r2_score

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

print("Start run")
TrackingClient.enable_system_metrics_logging()
with TrackingClient.start_run():
    TrackingClient.autolog()

    # Set the run name
    TrackingClient.set_run_name("Run Scikit-Learn 1")

    # Create and train the RandomForestRegressor model.
    rf = RandomForestRegressor(n_estimators=10, max_depth=6, max_features=3)
    rf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = rf.predict(X_test)

    # Calculate and log Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    TrackingClient.log_metric("mse", mse)

    # Calculate and log R-squared
    r2 = r2_score(y_test, y_pred)
    TrackingClient.log_metric("r2", r2)

    signature = TrackingClient.infer_signature(X_train, y_train)
    TrackingClient.sklearn.log_model(rf, "model", signature=signature)
