from pathlib import Path

from oip_tracking_client.v2.api.tracking import TrackingApi
from oip_tracking_client.v2.tracking import TrackingClient

# give absolute path
store = Path("./offline-tracking-store").resolve()

tc = TrackingClient(
    path_to_storage=str(store),
    offline_mode=True,
)

tc.set_experiment("v2-offline-demo")

# Workaround for oip-tracking-client==1.0.0:
# offline mode still tries to resolve run names through the tracking server.
# In offline mode, bypass that lookup and let TrackingClient create the
# local run directly.
TrackingApi.get_runs_for_experiment = lambda self, **kwargs: []

with tc.start_run(run_name="offline-run", log_system_metrics=False):
    tc.log_param("mode", "offline")
    tc.log_param("model", "demo-model")

    tc.log_metric("score", 0.82, step=1)
    tc.log_metric("loss", 0.41, step=1)

    output_file = store / "demo-output.txt"
    output_file.write_text("offline artifact demo\n")

    tc.log_artifact(str(output_file))

print(f"Offline tracking data written under: {store.resolve()}")