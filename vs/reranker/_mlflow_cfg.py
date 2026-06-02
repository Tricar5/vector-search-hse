"""Shared MLflow environment configuration."""
import os


def configure_mlflow(
    tracking_uri: str,
    s3_endpoint_url: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
) -> None:
    import mlflow

    os.environ['MLFLOW_S3_ENDPOINT_URL'] = s3_endpoint_url
    os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key_id
    os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_access_key
    mlflow.set_tracking_uri(tracking_uri)
