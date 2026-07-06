"""
Simple singleton state for multimodal text-to-image retrieval model.
"""
from typing import Any, Dict

_state: Dict[str, Any] = {}


def set_multimodal_state(**kwargs) -> None:
    _state.update(kwargs)


def get_multimodal_state() -> Dict[str, Any]:
    return _state


def clear_multimodal_state() -> None:
    _state.clear()


def is_multimodal_ready() -> bool:
    return "model" in _state and "tokenizer" in _state and "gallery_embeddings" in _state
