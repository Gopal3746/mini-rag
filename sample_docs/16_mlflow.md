# Experiment Tracking Convention

Retrieval experiments are logged to MLflow. Each run records embedding model, chunk size, overlap, top-k value, prompt version, recall at k, and mean reciprocal rank. The selected configuration is registered with the alias champion so downstream services can discover the approved version.
