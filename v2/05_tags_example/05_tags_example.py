import os

from oip_tracking_client.v2.tracking import TrackingClient


api_host = os.getenv("OICM_TRACKING_API_HOST", "http://localhost:8082/api/tracking")
api_key = os.getenv("OICM_TRACKING_API_KEY", "<api_key>")
workspace_id = os.getenv("OICM_WORKSPACE_ID", "<workspace_id>")


tc = TrackingClient(api_host=api_host, api_key=api_key)
tc.set_experiment(
    experiment_name="v2-tags-demo",
    workspace_id=workspace_id,
)
tc.set_experiment_tags(["demo", "tracking-client", "v2"])

with tc.start_run(
    run_name="tagged-run",
    tags=["candidate", "review"],
    log_system_metrics=False,
):
    tc.set_run_tags(["manual-metrics", "baseline"])
    tc.log_param("model", "ridge")
    tc.log_metric("score", 0.82)
