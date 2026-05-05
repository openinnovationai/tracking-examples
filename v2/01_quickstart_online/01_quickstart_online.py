import os

from oip_tracking_client.v2.tracking import TrackingClient


api_host = os.getenv("OICM_TRACKING_API_HOST", "http://localhost:8082/api/tracking")
api_key = os.getenv("OICM_TRACKING_API_KEY", "<api_key>")
workspace_id = os.getenv("OICM_WORKSPACE_ID", "<workspace_id>")


tc = TrackingClient(api_host=api_host, api_key=api_key)

tc.set_experiment(
    experiment_name="v2-quickstart-online",
    workspace_id=workspace_id,
)

with tc.start_run(run_name="smoke_test", log_system_metrics=False):
    tc.log_param("model", "xgboost")
    tc.log_metric("auc", 0.91)
