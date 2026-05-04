import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings


REPO_ROOT = Path(__file__).resolve().parents[2]
RATE_LIMIT_DEFAULT = "10/minute"


load_dotenv(REPO_ROOT / ".env", override=False)


def _resolve_existing_path(env_name: str, default_path: str, fallback_path: Path) -> str:
    candidate = Path(os.getenv(env_name, default_path))
    if candidate.exists():
        return str(candidate)

    if fallback_path.exists():
        return str(fallback_path)

    return str(candidate)


class Settings(BaseSettings):
    """
    Aplikasi settings dengan dukungan environment variables.
    """
    
    # App settings
    APP_NAME: str = "Batik Digital Gallery - FastAPI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    WORKERS: int = 1  # WAJIB 1 worker, sesuai spesifikasi untuk batasan RAM
    TIMEOUT_KEEP_ALIVE: int = 30
    
    # API security
    API_KEY: str = os.getenv("API_KEY", "your-secret-api-key")
    
    # Model settings
    MODEL_PATH: str = _resolve_existing_path("MODEL_PATH", "/app/models", REPO_ROOT / "models")
    MODELS_DIR_EXISTS: bool = Path(MODEL_PATH).exists()
    DATA_PATH: str = _resolve_existing_path("DATA_PATH", "/app/data", REPO_ROOT / "data")
    CHECKPOINTS_PATH: str = _resolve_existing_path("CHECKPOINTS_PATH", "/app/checkpoints", REPO_ROOT / "checkpoints")
    TPU_PATH: str = _resolve_existing_path("TPU_PATH", "/app/tpu", REPO_ROOT / "tpu")

    MOTIF_MODEL_FILE: str = os.getenv(
        "MOTIF_MODEL_FILE",
        "augmentTest_batik_cnn_pararel_elu3.h5",
    )
    MOTIF_LABEL_FILE: str = os.getenv(
        "MOTIF_LABEL_FILE",
        "label_mapping_pararelEluAugment3.json",
    )
    TULIS_MODEL_FILE: str = os.getenv(
        "TULIS_MODEL_FILE",
        "model_ConvNextTiny_original_all.pt",
    )
    CBIR_FEATURES_FILE: str = os.getenv(
        "CBIR_FEATURES_FILE",
        "features_768_features.npy",
    )
    CBIR_KMEANS_FILE: str = os.getenv(
        "CBIR_KMEANS_FILE",
        "features_768_kmeans_model.pkl",
    )
    CBIR_INDEX_FILE: str = os.getenv(
        "CBIR_INDEX_FILE",
        "features_768_indexed_database.csv",
    )
    CBIR_FEATURE_EXTRACTOR_WEIGHTS: str = os.getenv(
        "CBIR_FEATURE_EXTRACTOR_WEIGHTS",
        "",
    )
    CBIR_TOP_K: int = int(os.getenv("CBIR_TOP_K", 5))

    # Color FAISS settings
    COLOR_FAISS_SCENARIO: str = os.getenv("COLOR_FAISS_SCENARIO", "s1")
    COLOR_FAISS_MAX_SIZE: int = int(os.getenv("COLOR_FAISS_MAX_SIZE", 384))
    COLOR_FAISS_CANDIDATE_MULTIPLIER: int = int(os.getenv("COLOR_FAISS_CANDIDATE_MULTIPLIER", 20))

    # S3 settings
    S3_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    S3_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    S3_BUCKET_NAME: str = os.getenv("AWS_BUCKET_NAME", "")
    S3_BUCKET_NAME_CBIR: str = os.getenv("AWS_BUCKET_NAME_CBIR", "")
    S3_BUCKET_NAME_COLOR_FAISS: str = os.getenv("AWS_BUCKET_NAME_COLOR_FAISS", "")
    S3_BATIK_FASHION_ROOT_URL: str = os.getenv(
        "S3_BATIK_FASHION_ROOT_URL",
        "https://is3.cloudhost.id/color-dominant-batik",
    )
    S3_ENDPOINT_URL: str = os.getenv("AWS_ENDPOINT_URL", "")
    S3_REGION: str = os.getenv("AWS_REGION", "")
    S3_DATASET_BASE_PATH: str = os.getenv("S3_DATASET_BASE_PATH", "")
    S3_PRESIGN_EXPIRES: int = int(os.getenv("S3_PRESIGN_EXPIRES", 3600))
    
    # Request settings
    MAX_IMAGE_SIZE_MB: int = int(os.getenv("MAX_IMAGE_SIZE_MB", 5))
    MAX_IMAGE_SIZE_BYTES: int = int(
        os.getenv("MAX_IMAGE_SIZE_BYTES", str(MAX_IMAGE_SIZE_MB * 1024 * 1024))
    )
    MIN_IMAGE_SIZE_BYTES: int = 1024  # 1 KB
    ALLOWED_IMAGE_FORMATS: list = ["JPEG", "PNG", "WEBP"]
    ALLOWED_CONTENT_TYPES: list = ["image/jpeg", "image/png", "image/webp"]

    # Session storage
    SESSIONS_PATH: str = os.getenv("SESSIONS_PATH", str(REPO_ROOT / "sessions"))
    
    # Timeout settings
    INFERENCE_TIMEOUT_SECONDS: int = 30
    
    # Rate limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_CLASSIFY: str = os.getenv("RATE_LIMIT_CLASSIFY", "10/minute")
    RATE_LIMIT_CBIR: str = os.getenv("RATE_LIMIT_CBIR", "5/minute")
    RATE_LIMIT_HEALTH: str = os.getenv("RATE_LIMIT_HEALTH", "60/minute")
    RATE_LIMIT_GLOBAL: str = os.getenv("RATE_LIMIT_GLOBAL", "100/minute")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"

    def model_post_init(self, __context) -> None:
        path_fallbacks = {
            "MODEL_PATH": REPO_ROOT / "models",
            "DATA_PATH": REPO_ROOT / "data",
            "CHECKPOINTS_PATH": REPO_ROOT / "checkpoints",
            "TPU_PATH": REPO_ROOT / "tpu",
        }

        for field_name, fallback_path in path_fallbacks.items():
            current_path = Path(getattr(self, field_name))
            if current_path.exists():
                continue
            if fallback_path.exists():
                object.__setattr__(self, field_name, str(fallback_path))

        object.__setattr__(self, "MODELS_DIR_EXISTS", Path(self.MODEL_PATH).exists())


# Global settings instance
settings = Settings()
