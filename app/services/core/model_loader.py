"""
Singleton model loader for efficient model management.
Models are loaded once at application startup and reused for all requests.
"""

import torch
import logging
from app.services.core.model import FeatureEncoder, RecoloringDecoder

logger = logging.getLogger(__name__)


class ModelLoader:
    """Singleton pattern for loading and managing PyTorch models"""
    
    _instance = None
    _FE = None
    _RD = None
    _device = None

    @classmethod
    def get_instance(cls):
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self, fe_path: str, rd_path: str, device: str):
        """
        Load model weights from disk. Called once at app startup.
        
        Args:
            fe_path: Path to FeatureEncoder weights
            rd_path: Path to RecoloringDecoder weights
            device: 'cuda' or 'cpu'
        
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            self._device = device
            logger.info(f"Loading models on device: {device}")
            
            # Load FeatureEncoder
            self._FE = FeatureEncoder()
            self._FE.load_state_dict(torch.load(fe_path, map_location=device))
            self._FE.eval().to(device)
            logger.info(f"✓ FeatureEncoder loaded from {fe_path}")
            
            # Load RecoloringDecoder
            self._RD = RecoloringDecoder()
            self._RD.load_state_dict(torch.load(rd_path, map_location=device))
            self._RD.eval().to(device)
            logger.info(f"✓ RecoloringDecoder loaded from {rd_path}")
            
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
            raise RuntimeError(f"Model file not found: {e}")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise RuntimeError(f"Failed to load models: {e}")

    def get_models(self) -> tuple:
        """
        Get loaded models.
        
        Returns:
            Tuple of (FeatureEncoder, RecoloringDecoder, device_string)
        
        Raises:
            RuntimeError: If models haven't been loaded yet
        """
        if self._FE is None or self._RD is None:
            raise RuntimeError("Models not loaded. Call load() first.")
        return self._FE, self._RD, self._device

    @property
    def is_ready(self) -> bool:
        """Check if models are loaded and ready"""
        return self._FE is not None and self._RD is not None
