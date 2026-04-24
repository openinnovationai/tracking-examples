import os

from oip_tracking_client.v2.tracking import TrackingClient


api_host = os.getenv("OICM_TRACKING_API_HOST", "http://localhost:8082/api/tracking")
api_key = os.getenv("OICM_TRACKING_API_KEY", "<api_key>")
workspace_id = os.getenv("OICM_WORKSPACE_ID", "<workspace_id>")


tc = TrackingClient(api_host=api_host, api_key=api_key)

tc.set_experiment("v2-switch-exp-a", workspace_id=workspace_id)
with tc.start_run(run_name="run-in-a", log_system_metrics=False):
    tc.log_metric("m1", 1.0)

# Switch experiments only after the previous run has ended.
tc.set_experiment("v2-switch-exp-b", workspace_id=workspace_id)
with tc.start_run(run_name="run-in-b", log_system_metrics=False):
    tc.log_metric("m1", 2.0)
