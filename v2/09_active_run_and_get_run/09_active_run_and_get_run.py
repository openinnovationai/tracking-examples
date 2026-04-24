import os

from oip_tracking_client.v2.tracking import TrackingClient


api_host = os.getenv("OICM_TRACKING_API_HOST", "http://localhost:8082/api/tracking")
api_key = os.getenv("OICM_TRACKING_API_KEY", "<api_key>")
workspace_id = os.getenv("OICM_WORKSPACE_ID", "<workspace_id>")


tc = TrackingClient(api_host=api_host, api_key=api_key)
tc.set_experiment(
    experiment_name="v2-active-run-and-get-run",
    workspace_id=workspace_id,
)

with tc.start_run(run_name="lookup-demo", log_system_metrics=False):
    tc.log_param("model", "sgd")
    active = tc.active_run()
    print("Active run id:", active.info.run_id)

latest = tc.last_active_run()
fetched = tc.get_run(latest.info.run_id)

print("Last active run id:", latest.info.run_id)
print("Fetched run status:", fetched.info.status)
