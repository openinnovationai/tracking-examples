from fastai.vision.all import (
    URLs,
    untar_data,
    ImageDataLoaders,
    vision_learner,
    resnet18,
    accuracy,
)
import numpy as np
from oip_tracking_client.tracking import TrackingClient

api_host = "http://<oi_platform>/api"

api_key = "<api_key>"

workspace_name = "<workspace_name>"

experiment_name = "<experiment_name>"

# set up TrackingClient
TrackingClient.connect(api_host, api_key, workspace_name)

# set the experiment
TrackingClient.set_experiment(experiment_name)


path = untar_data(URLs.CIFAR)
dls = ImageDataLoaders.from_folder(path, train="train", valid="test")
# Get a batch of validation data
inputs, targets = dls.valid.one_batch()

# Extract the first validation data item
x_valid = inputs.cpu().numpy()
_, ch, w, h = x_valid.shape
x_valid = np.zeros(
    shape=(1, w, h, ch), dtype=np.uint8
)  # Could be any Numpy array with shape supported by PIL.Image.from_array

learn = vision_learner(dls, resnet18, metrics=accuracy)


with TrackingClient.start_run() as run:

    # Set the run name
    TrackingClient.set_run_name("Run FastAI 1")

    # Log the parameters
    TrackingClient.log_params({"epochs": 2, "lr": 1e-3})

    # Train the model
    learn.fit_one_cycle(1, lr_max=1e-3)

    # Log the trained model
    signature = TrackingClient.infer_signature(
        x_valid, np.zeros(shape=(1, 10), dtype=np.float32)
    )
    TrackingClient.fastai.log_model(learn, "model", signature=signature)
