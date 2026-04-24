import numpy as np
import statsmodels.api as sm

from oip_tracking_client.tracking import TrackingClient

api_host = "http://<oi_platform>/api"

api_key = "<api_key>"

workspace_name = "<workspace_name>"

experiment_name = "<experiment_name>"


# set up TrackingClient
TrackingClient.connect(api_host, api_key, workspace_name)

# set the experiment
TrackingClient.set_experiment(experiment_name)

# Generate sample data with shape (-1, 10) for 10 features and a single output
np.random.seed(42)
num_samples = 100
num_features = 10

X = np.random.rand(num_samples, num_features)
# Generating the response variable as a linear combination of the features with some noise
true_coefficients = np.random.rand(num_features)
noise = np.random.normal(loc=0, scale=0.1, size=num_samples)
y = np.dot(X, true_coefficients) + noise

# Add a constant term to the independent variable (intercept)
X = sm.add_constant(X)

TrackingClient.enable_system_metrics_logging()
with TrackingClient.start_run():
    TrackingClient.autolog()

    # Set the run name
    TrackingClient.set_run_name("Test Statsmodel 1")

    # Create a linear regression model
    model = sm.OLS(y, X)

    # Fit the model to the data
    model_fit = model.fit()

    # Log the model using mlflow.statsmodels.log_model
    signature = TrackingClient.infer_signature(X, y)
    TrackingClient.statsmodels.log_model(model_fit, "model", signature=signature)
