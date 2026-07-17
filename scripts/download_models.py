"""
Script ini dijalankan saat container startup untuk mendownload
folder models/ dan data/ dari HF Model Repo ke dalam container.

Repo: galeriBatikMalang/ML_models
"""
import os
import sys
from pathlib import Path


def download_models():
    try:
        from huggingface_hub import snapshot_download, hf_hub_download
    except ImportError:
        print("[download_models] ERROR: huggingface_hub tidak terinstall!")
        sys.exit(1)

    repo_id = "galeriBatikMalang/ML_models"
    token = os.getenv("HF_TOKEN")
    app_dir = Path(__file__).resolve().parent.parent  # root app

    models_dir = app_dir / "models"
    data_dir = app_dir / "data"

    # ── Download models/ ──
    if models_dir.exists() and (any(models_dir.rglob("*.h5")) or any(models_dir.rglob("*.pt"))):
        print("[download_models] ✅ Models sudah ada, skip download models/.")
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

    # ── Download data/ ──
    if data_dir.exists() and any(data_dir.iterdir()):
        print("[download_models] ✅ Data sudah ada, skip download data/.")
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

    # ── Download UNet INT8 jika tersedia di HF Repo ──
    # File ini dihasilkan dari: python scripts/quantize_unet.py
    # Jauh lebih kecil: ~820 MB vs ~3,200 MB (FP32)
    int8_path = models_dir / "colorizer" / "unet" / "unet_int8_state_dict.pt"
    if int8_path.exists():
        print("[download_models] ✅ UNet INT8 sudah ada, skip.")
    else:
        print("[download_models] ⬇️  Mencoba download UNet INT8 (opsional, ~820 MB)...")
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename="models/colorizer/unet/unet_int8_state_dict.pt",
                local_dir=str(app_dir),
                token=token,
            )
            print("[download_models] ✅ UNet INT8 berhasil didownload.")
        except Exception as e:
            print(f"[download_models] ℹ️  UNet INT8 tidak ada di repo (akan fallback ke FP32): {e}")
            print("[download_models]    Jalankan `python scripts/quantize_unet.py` lalu upload ke HF Repo.")


if __name__ == "__main__":
    download_models()
