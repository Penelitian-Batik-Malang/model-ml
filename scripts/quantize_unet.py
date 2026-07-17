"""
Script quantisasi UNet ke INT8 untuk deployment CPU dengan RAM terbatas.

Jalankan SEKALI secara lokal sebelum upload ke HF:
    python scripts/quantize_unet.py

Output: models/colorizer/unet/unet_int8_state_dict.pt (~820 MB)
Dari   : models/colorizer/unet/diffusion_pytorch_model.safetensors (~3.2 GB)

Penghematan RAM saat deployment: ~6.4 GB → ~1.6 GB (INT8 di CPU)
Catatan: Kualitas sedikit turun, biasanya tidak terlihat secara visual.
"""
import sys
import time
from pathlib import Path

import torch
import torch.ao.quantization as tq
from diffusers import UNet2DConditionModel

# ── Path ──
ROOT = Path(__file__).resolve().parent.parent
UNET_DIR = ROOT / "models" / "colorizer" / "unet"
OUT_PATH = UNET_DIR / "unet_int8_state_dict.pt"


def main():
    if not UNET_DIR.exists():
        print(f"[ERROR] UNet directory tidak ditemukan: {UNET_DIR}")
        sys.exit(1)

    if OUT_PATH.exists():
        answer = input(f"[WARNING] {OUT_PATH} sudah ada. Overwrite? (y/N): ").strip().lower()
        if answer != "y":
            print("Dibatalkan.")
            sys.exit(0)

    print(f"[quantize_unet] Loading UNet FP32 dari: {UNET_DIR}")
    print("[quantize_unet] Ini membutuhkan ~6.4 GB RAM dan beberapa menit...")
    t0 = time.time()

    unet = UNet2DConditionModel.from_pretrained(str(UNET_DIR), torch_dtype=torch.float32)
    unet.eval()
    print(f"[quantize_unet] UNet FP32 loaded dalam {time.time() - t0:.1f}s")

    # ── Dynamic INT8 Quantization ──
    # Hanya quantisasi nn.Linear (attention, FFN) karena:
    # - nn.Conv2d kadang memberikan error shape di INT8 dynamic quant
    # - nn.Linear menyumbang mayoritas bobot di UNet SD v1.5
    print("[quantize_unet] Menerapkan Dynamic INT8 Quantization pada nn.Linear...")
    t1 = time.time()
    unet_int8 = tq.quantize_dynamic(
        unet,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    print(f"[quantize_unet] Quantisasi selesai dalam {time.time() - t1:.1f}s")

    # ── Simpan state dict ──
    print(f"[quantize_unet] Menyimpan ke: {OUT_PATH}")
    t2 = time.time()
    torch.save(unet_int8.state_dict(), str(OUT_PATH))
    size_mb = OUT_PATH.stat().st_size / (1024 ** 2)
    print(f"[quantize_unet] ✅ Tersimpan dalam {time.time() - t2:.1f}s")
    print(f"[quantize_unet] Ukuran file: {size_mb:.1f} MB (dari ~3,200 MB)")
    print(f"[quantize_unet] Total waktu: {time.time() - t0:.1f}s")
    print()
    print("Langkah berikutnya:")
    print("  1. Upload unet_int8_state_dict.pt ke HF Model Repo (galeriBatikMalang/ML_models)")
    print("     huggingface-cli upload galeriBatikMalang/ML_models \\")
    print("         models/colorizer/unet/unet_int8_state_dict.pt \\")
    print("         models/colorizer/unet/unet_int8_state_dict.pt")
    print("  2. Update scripts/download_models.py untuk ikut download file ini")


if __name__ == "__main__":
    main()
