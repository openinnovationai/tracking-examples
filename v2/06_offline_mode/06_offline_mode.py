from pathlib import Path

from oip_tracking_client.v2.tracking import TrackingClient

if __name__ == "__main__":
    store = Path("./offline-tracking-store").resolve()

    tc = TrackingClient(
        path_to_storage=str(store),
        offline_mode=True,
    )

    tc.set_experiment("v2-offline-demo")

    with tc.start_run(log_system_metrics=False):
        tc.log_param("mode", "offline")
        tc.log_param("model", "demo-model")
        tc.log_param("run_label", "offline-run")

        tc.log_metric("score", 0.82, step=1)
        tc.log_metric("loss", 0.41, step=1)

        output_file = store / "demo-output.txt"
        output_file.write_text("offline artifact demo\n")

    print(f"Offline tracking data written under: {store}")