import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import reuters
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.preprocessing.text import Tokenizer
from oip_tracking_client.tracking import TrackingClient

api_host = "http://<oi_platform>/api"

api_key = "<api_key>"

workspace_name = "<workspace_name>"

experiment_name = "<experiment_name>"


# set up TrackingClient
TrackingClient.connect(api_host, api_key, workspace_name)

# set the experiment
TrackingClient.set_experiment(experiment_name)


max_words = 50
batch_size = 32
epochs = 4

# Loading data
(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=max_words, test_split=0.2
)

num_classes = np.max(y_train) + 1

# Vectorizing sequence data
tokenizer = Tokenizer(num_words=max_words)
x_train = tokenizer.sequences_to_matrix(x_train, mode="binary")
x_test = tokenizer.sequences_to_matrix(x_test, mode="binary")

# Convert class vector to binary class matrix (for use with categorical_crossentropy)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

with TrackingClient.start_run():

    # Set the run name
    TrackingClient.set_run_name("Test Keras 1")

    # Building model
    model = Sequential()
    model.add(Dense(512, input_shape=(max_words,)))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_split=0.1,
    )

    # Log the model using mlflow.statsmodels.log_model
    signature = TrackingClient.infer_signature(x_train, y_train)
    TrackingClient.keras.log_model(model, "model", signature=signature)
