import logging

import boto3
from botocore.config import Config

from app.config.settings import settings

logger = logging.getLogger(__name__)

s3_client = None

try:
    if settings.S3_ACCESS_KEY_ID and settings.S3_SECRET_ACCESS_KEY and settings.S3_ENDPOINT_URL:
        s3_client = boto3.client(
            's3',
            endpoint_url=settings.S3_ENDPOINT_URL,
            aws_access_key_id=settings.S3_ACCESS_KEY_ID,
            aws_secret_access_key=settings.S3_SECRET_ACCESS_KEY,
            region_name=settings.S3_REGION,
            config=Config(signature_version='s3v4'),
        )
        logger.info("S3 client initialized for signature drive: %s", settings.AWS_BUCKET_SIGNATURE_DRIVE)
except Exception as e:
    logger.error("Failed to initialize S3 client: %s", e)


def get_s3_presigned_url(filename: str) -> str:
    """Generate a presigned URL for a file in the signature drive bucket."""
    if not s3_client or not settings.AWS_BUCKET_SIGNATURE_DRIVE:
        return ""
    try:
        url = s3_client.generate_presigned_url(
            ClientMethod='get_object',
            Params={
                'Bucket': settings.AWS_BUCKET_SIGNATURE_DRIVE,
                'Key': filename,
            },
            ExpiresIn=settings.S3_PRESIGN_EXPIRES,
        )
        return url
    except Exception as e:
        logger.error("Error generating presigned URL for %s: %s", filename, e)
        return ""
