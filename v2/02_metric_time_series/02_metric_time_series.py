import os

from sklearn.datasets import load_diabetes
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from oip_tracking_client.v2.tracking import TrackingClient


api_host = os.getenv("OICM_TRACKING_API_HOST", "http://localhost:8082/api/tracking")
api_key = os.getenv("OICM_TRACKING_API_KEY", "<api_key>")
workspace_id = os.getenv("OICM_WORKSPACE_ID", "<workspace_id>")


def rmse(y_true, y_pred) -> float:
    return mean_squared_error(y_true, y_pred) ** 0.5


tc = TrackingClient(api_host=api_host, api_key=api_key)
tc.set_experiment(
    experiment_name="v2-metric-time-series",
    workspace_id=workspace_id,
)

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

with tc.start_run(run_name="sgd-training-demo", log_system_metrics=False):
    tc.log_params({"dataset": "diabetes", "epochs": 8, "eta0": 0.01})

    model = SGDRegressor(
        loss="squared_error",
        learning_rate="constant",
        eta0=0.01,
        max_iter=1,
        tol=None,
        warm_start=True,
        random_state=42,
    )

    for step in range(1, 9):
        model.partial_fit(X_train, y_train)
        tc.log_metric("rmse", rmse(y_test, model.predict(X_test)), step=step)
