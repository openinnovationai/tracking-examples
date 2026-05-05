import torch
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    pipeline,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from oip_tracking_client.tracking import TrackingClient

api_host = "http://<oi_platform>/api"

api_key = "<api_key>"

workspace_name = "<workspace_name>"

experiment_name = "<experiment_name>"

# set up TrackingClient
TrackingClient.connect(api_host, api_key, workspace_name)

# set the experiment
TrackingClient.set_experiment(experiment_name)


# Load pre-trained BERT model and tokenizer
# model_name = "bert-base-uncased"
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertForSequenceClassification.from_pretrained(model_name)

# sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
model_name = "hakonmh/sentiment-xdistil-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Define a sample movie review
review = "I really enjoyed watching this movie."

# Tokenize the review and convert to tensor
inputs = tokenizer(
    review, return_tensors="pt", padding=True, truncation=True, max_length=128
)

input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

expected_label = 0


# Start MLflow run
with TrackingClient.start_run() as run:

    # Set the run name
    TrackingClient.set_run_name("Run Transformers 3")

    # Log parameters
    TrackingClient.log_param("model_name", model_name)

    # Perform sentiment analysis
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    # Get the predicted label
    predicted_label = torch.argmax(outputs.logits).item()
    sentiment = "positive" if predicted_label == 1 else "negative"

    TrackingClient.log_metric("delta", predicted_label - expected_label)

    TrackingClient.transformers.log_model(sentiment_analysis, "model")
