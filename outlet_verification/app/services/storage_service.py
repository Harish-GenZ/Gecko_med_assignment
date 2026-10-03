import io
import logging
import mimetypes
import re
import uuid
from typing import BinaryIO
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from PIL import Image

from app.config import get_settings

logger = logging.getLogger("outlet_verification.storage")

class StorageService:
    """
    Manages image storage in Railway S3-compatible Object Storage (Tigris).
    Provides upload, presigned URL generation, and download capabilities.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.endpoint_url = self.settings.ENDPOINT_URL or "https://t3.storageapi.dev"
        self.bucket_name = self.settings.Bucket_Name
        self.region = self.settings.Region if self.settings.Region != "auto" else "us-east-1"
        self.access_key = self.settings.ACCESS_KEY_ID
        self.secret_key = self.settings.SECRET_ACCESS_KEY

        self.s3_client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region,
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        )

    def upload_outlet_image(
        self,
        image_data: bytes | BinaryIO,
        filename: str = "storefront.jpg",
        content_type: str | None = None,
    ) -> tuple[str, str]:
        """
        Uploads an outlet photograph to Railway Object Storage.
        Returns:
            tuple[str, str]: (access_url, object_key)
        """
        if not self.bucket_name:
            raise ValueError("Bucket_Name is not configured in environment settings.")

        # Read bytes if BinaryIO
        if hasattr(image_data, "read"):
            data_bytes = image_data.read()
        else:
            data_bytes = image_data

        if not data_bytes:
            raise ValueError("Cannot upload empty image data.")

        # Sanitize filename
        safe_filename = re.sub(r"[^a-zA-Z0-9_.-]", "_", filename).lower()
        if not safe_filename:
            safe_filename = "storefront.jpg"

        # Auto-detect or default content type
        if not content_type:
            guessed_type, _ = mimetypes.guess_type(safe_filename)
            content_type = guessed_type or "image/jpeg"

        unique_id = uuid.uuid4().hex[:12]
        object_key = f"medical_shops/{unique_id}_{safe_filename}"

        logger.info(
            "Uploading %d bytes to S3 bucket '%s' with key '%s'...",
            len(data_bytes),
            self.bucket_name,
            object_key,
        )

        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=object_key,
                Body=data_bytes,
                ContentType=content_type,
            )
            logger.info("Successfully uploaded object to bucket: %s", object_key)

            # Generate long-lived presigned URL (valid for 7 days / 604800 seconds)
            presigned_url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": object_key},
                ExpiresIn=604800,
            )

            return presigned_url, object_key

        except ClientError as exc:
            logger.error("Failed to upload image to S3 bucket: %s", exc)
            raise RuntimeError(f"Storage upload failed: {exc}") from exc

    def get_presigned_url(self, object_key: str, expires_in: int = 604800) -> str:
        """
        Generates a presigned read URL for an existing object key.
        """
        try:
            return self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": object_key},
                ExpiresIn=expires_in,
            )
        except ClientError as exc:
            logger.error("Failed to generate presigned URL for key '%s': %s", object_key, exc)
            raise RuntimeError(f"Presigned URL generation failed: {exc}") from exc

    def get_object_bytes(self, object_key: str) -> tuple[bytes, str]:
        """
        Downloads the raw object bytes and content type from the bucket.
        """
        try:
            resp = self.s3_client.get_object(Bucket=self.bucket_name, Key=object_key)
            content = resp["Body"].read()
            content_type = resp.get("ContentType", "image/jpeg")
            return content, content_type
        except ClientError as exc:
            logger.error("Failed to fetch object '%s' from S3: %s", object_key, exc)
            raise RuntimeError(f"Storage fetch failed: {exc}") from exc


_storage_service_instance: StorageService | None = None

def get_storage_service() -> StorageService:
    global _storage_service_instance
    if _storage_service_instance is None:
        _storage_service_instance = StorageService()
    return _storage_service_instance
