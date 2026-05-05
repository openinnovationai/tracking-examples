# Tracking Examples (`v1`)

These are the original tracking examples that use the legacy client surface:

```python
from oip_tracking_client.tracking import TrackingClient
```

Most of these examples focus on library-specific model logging and older artifact-oriented flows.

## How To Run

Each library folder has its own dependencies:

```bash
cd v1/sklearn
pip install -r requirements.txt
python test_sklearn.py
```

Before running, update the configuration block inside the script:

```python
api_host = "http://<oi_platform>/api"
api_key = "<api_key>"
workspace_name = "<workspace_name>"
experiment_name = "<experiment_name>"
```
