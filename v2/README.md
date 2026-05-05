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

fill the below env vars

```
export OICM_TRACKING_API_HOST=<host_name_ending_with_api/tracking>
export OICM_TRACKING_API_KEY=<generated_from_api_key>
export OICM_WORKSPACE_ID=<workspace_id>
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
