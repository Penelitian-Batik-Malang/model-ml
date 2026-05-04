import logging
from pathlib import Path
from typing import Dict, Optional

from app.services.motif_classifier import MotifClassifier
from app.services.tulis_classifier import TulisClassifier
from app.services.cbir_engine import CBIREngine

logger = logging.getLogger(__name__)


class ModelLoader:
    """Singleton loader for motif, tulis, and CBIR models."""

    _instance: Optional["ModelLoader"] = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.motif_classifier: Optional[MotifClassifier] = None
        self.tulis_classifier: Optional[TulisClassifier] = None
        self.cbir_engine: Optional[CBIREngine] = None

        self.model_path: Optional[str] = None
        self.data_path: Optional[str] = None
        self.checkpoints_path: Optional[str] = None
        self.tpu_path: Optional[str] = None

        self.status: Dict[str, bool] = {"motif": False, "tulis": False, "cbir": False}
        self._initialized = True
        logger.info("ModelLoader initialized")

    def _resolve_file(self, primary_dir: Path, filename: str, fallback_dirs: list) -> Path:
        for base_dir in [primary_dir] + fallback_dirs:
            candidate = Path(base_dir) / filename
            if candidate.exists():
                return candidate
        return Path(primary_dir) / filename

    def load_model(
        self,
        model_path: str = "/app/models",
        data_path: str = "/app/data",
        checkpoints_path: str = "/app/checkpoints",
        tpu_path: str = "/app/tpu",
    ) -> bool:
        """Load all models and supporting data."""
        from app.config.settings import settings

        self.status = {"motif": False, "tulis": False, "cbir": False}

        self.model_path = model_path
        self.data_path = data_path
        self.checkpoints_path = checkpoints_path
        self.tpu_path = tpu_path

        model_dir = Path(model_path)
        data_dir = Path(data_path)
        checkpoints_dir = Path(checkpoints_path)

        motif_model = self._resolve_file(
            model_dir,
            settings.MOTIF_MODEL_FILE,
            [checkpoints_dir, data_dir],
        )
        motif_label = self._resolve_file(
            model_dir,
            settings.MOTIF_LABEL_FILE,
            [checkpoints_dir, data_dir],
        )

        self.motif_classifier = MotifClassifier(str(motif_model), str(motif_label))
        self.status["motif"] = self.motif_classifier.load()

        tulis_model = self._resolve_file(
            model_dir,
            settings.TULIS_MODEL_FILE,
            [checkpoints_dir, data_dir],
        )
        self.tulis_classifier = TulisClassifier(str(tulis_model), device="cpu")
        self.status["tulis"] = self.tulis_classifier.load()

        cbir_features = self._resolve_file(
            data_dir,
            settings.CBIR_FEATURES_FILE,
            [model_dir, checkpoints_dir],
        )
        cbir_kmeans = self._resolve_file(
            data_dir,
            settings.CBIR_KMEANS_FILE,
            [model_dir, checkpoints_dir],
        )
        cbir_index = self._resolve_file(
            data_dir,
            settings.CBIR_INDEX_FILE,
            [model_dir, checkpoints_dir],
        )
        cbir_weights = None
        if settings.CBIR_FEATURE_EXTRACTOR_WEIGHTS:
            cbir_weights = self._resolve_file(
                model_dir,
                settings.CBIR_FEATURE_EXTRACTOR_WEIGHTS,
                [checkpoints_dir, data_dir],
            )

        self.cbir_engine = CBIREngine(
            features_path=str(cbir_features),
            kmeans_path=str(cbir_kmeans),
            indexed_db_path=str(cbir_index),
            weights_path=str(cbir_weights) if cbir_weights else None,
            device="cpu",
        )
        self.status["cbir"] = self.cbir_engine.load()

        return any(self.status.values())

    def is_model_loaded(self) -> bool:
        return all(self.status.values())

    def is_motif_loaded(self) -> bool:
        return self.status.get("motif", False)

    def is_tulis_loaded(self) -> bool:
        return self.status.get("tulis", False)

    def is_cbir_loaded(self) -> bool:
        return self.status.get("cbir", False)

    def get_motif_classifier(self) -> Optional[MotifClassifier]:
        return self.motif_classifier

    def get_tulis_classifier(self) -> Optional[TulisClassifier]:
        return self.tulis_classifier

    def get_cbir_engine(self) -> Optional[CBIREngine]:
        return self.cbir_engine

    def check_model_exists(self, model_path: str = "/app/models") -> bool:
        try:
            model_dir = Path(model_path)
            return model_dir.exists() and model_dir.is_dir()
        except Exception as exc:
            logger.error("Error checking model path: %s", exc)
            return False


_model_loader: Optional[ModelLoader] = None


def get_model_loader() -> ModelLoader:
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader
