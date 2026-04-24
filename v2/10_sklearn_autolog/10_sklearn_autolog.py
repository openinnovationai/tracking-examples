import os

from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from oip_tracking_client.v2.tracking import TrackingClient


api_host = os.getenv("OICM_TRACKING_API_HOST", "http://localhost:8082/api/tracking")
api_key = os.getenv("OICM_TRACKING_API_KEY", "<api_key>")
workspace_id = os.getenv("OICM_WORKSPACE_ID", "<workspace_id>")


tc = TrackingClient(api_host=api_host, api_key=api_key)
tc.set_experiment(
    experiment_name="v2-sklearn-autolog-demo",
    workspace_id=workspace_id,
)
tc.set_experiment_tags(["v2", "sklearn", "autolog"])

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with tc.start_run(run_name="random-forest-regressor", log_system_metrics=False):
    tc.set_run_tags(["baseline", "regression", "random-forest"])

    # sklearn autolog hooks into estimator.fit(...).
    # It captures supported training metadata automatically, so we only add
    # the explicit run context and the final comparison metrics we care about.
    tc.sklearn.autolog(log_models=False)

    tc.log_params({"dataset": "diabetes", "test_size": 0.2, "random_state": 42})

    model = RandomForestRegressor(
        n_estimators=40,
        max_depth=6,
        random_state=42,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    tc.log_metric("test_rmse", mean_squared_error(y_test, predictions) ** 0.5)
    tc.log_metric("test_r2", r2_score(y_test, predictions))
