import os

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError


def make_s3_client(cfg):
    return boto3.client(
        "s3",
        endpoint_url=cfg["endpoint_url"],
        aws_access_key_id=cfg["access_key"],
        aws_secret_access_key=cfg["secret_key"],
        config=Config(signature_version="s3v4"),
        region_name=cfg.get("region", "us-east-1"),
    )


def ensure_bucket(client, bucket):
    try:
        client.head_bucket(Bucket=bucket)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in ("404", "NoSuchBucket", "NotFound"):
            client.create_bucket(Bucket=bucket)
        else:
            raise


def upload_file(cfg, local_path, key):
    client = make_s3_client(cfg)
    ensure_bucket(client, cfg["bucket"])
    client.upload_file(local_path, cfg["bucket"], key)
    print(f"uploaded {local_path} -> s3://{cfg['bucket']}/{key}")


def download_file(cfg, key, local_path):
    client = make_s3_client(cfg)
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    client.download_file(cfg["bucket"], key, local_path)
    print(f"downloaded s3://{cfg['bucket']}/{key} -> {local_path}")
