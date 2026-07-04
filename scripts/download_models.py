"""
Script ini dijalankan saat container startup untuk mendownload
folder models/ dan data/ dari HF Model Repo ke dalam container.

Repo: Fadhlu/models_galeridigital-batikmalang
"""
import os
import sys
from pathlib import Path

def download_models():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[download_models] ERROR: huggingface_hub tidak terinstall!")
        sys.exit(1)

    repo_id = "Fadhlu/models_galeridigital-batikmalang"
    token = os.getenv("HF_TOKEN")
    app_dir = Path(__file__).resolve().parent.parent  # root app

    models_dir = app_dir / "models"
    data_dir = app_dir / "data"

    # Cek apakah models sudah ada (skip jika sudah)
    if models_dir.exists() and any(models_dir.rglob("*.h5")) or any(models_dir.rglob("*.pt")):
        print("[download_models] ✅ Models sudah ada, skip download.")
    else:
        print(f"[download_models] ⬇️  Mendownload models/ dari {repo_id}...")
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=["models/**"],
            local_dir=str(app_dir),
            token=token,
        )
        print("[download_models] ✅ Download models/ selesai.")

    if data_dir.exists() and any(data_dir.iterdir()):
        print("[download_models] ✅ Data sudah ada, skip download.")
    else:
        print(f"[download_models] ⬇️  Mendownload data/ dari {repo_id}...")
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=["data/**"],
            local_dir=str(app_dir),
            token=token,
        )
        print("[download_models] ✅ Download data/ selesai.")

if __name__ == "__main__":
    download_models()
