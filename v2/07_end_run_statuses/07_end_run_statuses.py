import os

from oip_tracking_client.v2.tracking import TrackingClient


api_host = os.getenv("OICM_TRACKING_API_HOST", "http://localhost:8082/api/tracking")
api_key = os.getenv("OICM_TRACKING_API_KEY", "<api_key>")
workspace_id = os.getenv("OICM_WORKSPACE_ID", "<workspace_id>")


tc = TrackingClient(api_host=api_host, api_key=api_key)
tc.set_experiment(
    experiment_name="v2-end-run-statuses",
    workspace_id=workspace_id,
)

tc.start_run(run_name="finished-run", log_system_metrics=False)
tc.log_metric("score", 0.90)
tc.end_run(status="FINISHED")

tc.start_run(run_name="failed-run", log_system_metrics=False)
tc.log_param("failure_reason", "simulated")
tc.end_run(status="FAILED")

tc.start_run(run_name="killed-run", log_system_metrics=False)
tc.log_param("cancel_reason", "manual-stop")
tc.end_run(status="KILLED")
