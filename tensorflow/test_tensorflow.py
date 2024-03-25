import tensorflow as tf

from oip_tracking_client.tracking import TrackingClient

api_host = "http://<oi_platform>/api"

api_key = "<api_key>"

workspace_name = "<workspace_name>"

experiment_name = "<experiment_name>"

# set up TrackingClient
TrackingClient.connect(api_host, api_key, workspace_name)

# set the experiment
TrackingClient.set_experiment(experiment_name)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10),
    ]
)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

with TrackingClient.start_run():

    # Set the run name
    TrackingClient.set_run_name("Experiment Tensorflow 1")

    model.fit(x_train, y_train, epochs=1)

    signature = TrackingClient.infer_signature(x_train, y_train)
    TrackingClient.tensorflow.log_model(model, "model", signature=signature)
