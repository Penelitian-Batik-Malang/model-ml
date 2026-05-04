import logging
from typing import Optional

from app.config.fashion import FASHION_CBIR_FEATURES_NPZ
from app.services.fashion_recommendation_engine import load_batik_database

logger = logging.getLogger(__name__)

_fashion_cbir_db: Optional[dict] = None


def init_fashion_cbir_db() -> Optional[dict]:
    global _fashion_cbir_db
    if _fashion_cbir_db is not None:
        return _fashion_cbir_db

    if FASHION_CBIR_FEATURES_NPZ.exists():
        _fashion_cbir_db = load_batik_database(FASHION_CBIR_FEATURES_NPZ)
        logger.info("Fashion CBIR DB loaded: %s items", len(_fashion_cbir_db.get("filenames", [])))
    else:
        logger.warning("Fashion CBIR NPZ not found at %s", FASHION_CBIR_FEATURES_NPZ)

    return _fashion_cbir_db


def get_fashion_cbir_db() -> Optional[dict]:
    return _fashion_cbir_db
