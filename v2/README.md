# Tracking Examples (`v2`)

These examples use the current client API:

```python
from oip_tracking_client.v2.tracking import TrackingClient
```

Each example lives in its own folder. Every folder contains:

- one self-contained Python example
- one `requirements.txt` for that example

## Example Folders

- `01_quickstart_online`
- `02_metric_time_series`
- `03_run_resume_by_name`
- `04_run_resume_by_id`
- `05_tags_example`
- `06_offline_mode`
- `07_end_run_statuses`
- `08_switch_experiments_safely`
- `09_create_get_delete_experiment`
- `10_active_run_and_get_run`
- `11_sklearn_autolog`
- `12_xgboost_autolog`

## Before You Run

Install the internal `v2` tracking client in your Python environment first. The environment must provide:

```python
from oip_tracking_client.v2.tracking import TrackingClient
```

## How To Run One Example

Example:

```bash
cd v2/01_quickstart_online
pip install -r requirements.txt
python 01_quickstart_online.py
```

For an example with extra ML dependencies:

```bash
cd v2/10_sklearn_autolog
pip install -r requirements.txt
python 10_sklearn_autolog.py
```

## Notes

- `01` to `10` focus on `TrackingClient` workflows.
- `11` and `12` keep the library-flavored autolog examples.
- `06_offline_mode` stays on `TrackingClient` only and includes a small workaround for the current `oip-tracking-client==1.0.0` offline run-name lookup bug.
- `v1/` still contains the older artifact-heavy examples.
