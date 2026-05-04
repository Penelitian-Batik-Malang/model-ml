import logging
from typing import Optional

import boto3
from botocore.client import Config

from app.config.settings import settings

logger = logging.getLogger(__name__)


class S3Storage:
    def __init__(self) -> None:
        self.bucket = settings.S3_BUCKET_NAME
        self.bucket_cbir = settings.S3_BUCKET_NAME_CBIR or self.bucket
        self.bucket_color_faiss = settings.S3_BUCKET_NAME_COLOR_FAISS or self.bucket
        self.base_path = settings.S3_DATASET_BASE_PATH
        self.expires_in = settings.S3_PRESIGN_EXPIRES
        self.client = self._create_client()

    def resolve_bucket(self, bucket_name: Optional[str] = None) -> Optional[str]:
        bucket = bucket_name or self.bucket
        return bucket or None

    def _create_client(self):
        if not settings.S3_ACCESS_KEY_ID or not settings.S3_SECRET_ACCESS_KEY:
            logger.warning("S3 credentials not configured")
            return None

        return boto3.client(
            "s3",
            endpoint_url=settings.S3_ENDPOINT_URL or None,
            region_name=settings.S3_REGION or None,
            aws_access_key_id=settings.S3_ACCESS_KEY_ID,
            aws_secret_access_key=settings.S3_SECRET_ACCESS_KEY,
            config=Config(signature_version="s3v4"),
        )

    def normalize_key(self, image_path: str) -> str:
        if not image_path:
            return ""

        base = self.base_path or ""
        cleaned = image_path
        if base and cleaned.startswith(base):
            cleaned = cleaned[len(base):]
        cleaned = cleaned.lstrip("/\\")
        return cleaned.replace("\\", "/")

    def generate_presigned_url(self, key: str, bucket_name: Optional[str] = None) -> Optional[str]:
        bucket = self.resolve_bucket(bucket_name)
        if not self.client or not key or not bucket:
            return None
        try:
            return self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=self.expires_in,
            )
        except Exception as exc:
            logger.error("Failed to generate presigned URL: %s", exc)
            return None


_s3_storage: Optional[S3Storage] = None


def get_s3_storage() -> S3Storage:
    global _s3_storage
    if _s3_storage is None:
        _s3_storage = S3Storage()
    return _s3_storage
