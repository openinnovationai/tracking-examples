import os

from oip_tracking_client.v2.tracking import TrackingClient


api_host = os.getenv("OICM_TRACKING_API_HOST", "http://localhost:8082/api/tracking")
api_key = os.getenv("OICM_TRACKING_API_KEY", "<api_key>")
workspace_id = os.getenv("OICM_WORKSPACE_ID", "<workspace_id>")

run_name = "resume-by-name-demo"

tc = TrackingClient(api_host=api_host, api_key=api_key)
tc.set_experiment(
    experiment_name="v2-run-resume-by-name",
    workspace_id=workspace_id,
)

with tc.start_run(run_name=run_name, log_system_metrics=False):
    tc.log_param("phase", "first-pass")

# If a run with the same name already exists in the active experiment,
# start_run(run_name=...) continues that run instead of creating a new one.
with tc.start_run(run_name=run_name, log_system_metrics=False):
    tc.log_param("phase", "second-pass")
    tc.log_metric("score", 0.87)
