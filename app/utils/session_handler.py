import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

from app.config.settings import settings

SESSIONS_DIR = Path(settings.SESSIONS_PATH)


def ensure_sessions_dir() -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def get_session_dir(session_id: str) -> Path:
    ensure_sessions_dir()
    return SESSIONS_DIR / session_id


def init_session(session_id: str) -> Path:
    session_dir = get_session_dir(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    meta_path = session_dir / "meta.json"
    if not meta_path.exists():
        save_session_meta(session_id, {"parts_blended": [], "parts_detected": []})
    return session_dir


def session_exists(session_id: str) -> bool:
    return get_session_dir(session_id).is_dir()


def save_session_meta(session_id: str, meta: Dict[str, Any]) -> None:
    session_dir = get_session_dir(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    meta_path = session_dir / "meta.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")


def load_session_meta(session_id: str) -> Dict[str, Any]:
    session_dir = get_session_dir(session_id)
    meta_path = session_dir / "meta.json"
    if not meta_path.exists():
        return {"parts_blended": [], "parts_detected": []}
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        return {
            "parts_blended": data.get("parts_blended", []),
            "parts_detected": data.get("parts_detected", []),
        }
    except json.JSONDecodeError:
        return {"parts_blended": [], "parts_detected": []}


def set_detected_parts(session_id: str, parts: List[str]) -> None:
    meta = load_session_meta(session_id)
    meta["parts_detected"] = parts
    save_session_meta(session_id, meta)


def add_blended_part(session_id: str, part: str, instance_index: int = 0) -> None:
    meta = load_session_meta(session_id)
    blended = meta.get("parts_blended", [])
    if part not in blended:
        blended.append(part)
    meta["parts_blended"] = blended
    save_session_meta(session_id, meta)


def reset_blended_parts(session_id: str) -> None:
    meta = load_session_meta(session_id)
    meta["parts_blended"] = []
    save_session_meta(session_id, meta)


def cleanup_old_sessions(max_age_hours: int = 2) -> None:
    ensure_sessions_dir()
    now = time.time()
    for session_dir in SESSIONS_DIR.iterdir():
        if not session_dir.is_dir():
            continue
        try:
            age = now - session_dir.stat().st_mtime
            if age > max_age_hours * 3600:
                shutil.rmtree(session_dir)
        except OSError:
            continue
