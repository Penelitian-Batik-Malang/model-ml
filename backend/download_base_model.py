# ============================================================
# DOWNLOAD BASE MODEL COMPONENTS FOR OFFLINE DEPLOYMENT
# ============================================================
import os
import sys

# Tambahkan folder backend ke path pencarian modul jika dipanggil dari luar
sys.path.insert(0, os.path.dirname(__file__))

from diffusers import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel

# Gunakan model dasar stable diffusion v1-5
BASE_MODEL = "runwayml/stable-diffusion-v1-5"

# Tentukan folder output lokal
LOCAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "stable-diffusion-v1-5"))

def main():
    print("Memulai pengunduhan model dasar stable diffusion v1-5 ke folder lokal:")
    print("Target folder: " + LOCAL_DIR + "\n")
    
    os.makedirs(LOCAL_DIR, exist_ok=True)
    
    # 1. Download & Save Tokenizer
    print("1/3. Mengunduh dan menyimpan Tokenizer...")
    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL, subfolder="tokenizer")
    tokenizer.save_pretrained(os.path.join(LOCAL_DIR, "tokenizer"))
    print("[OK] Tokenizer berhasil disimpan.\n")
    
    # 2. Download & Save Text Encoder
    print("2/3. Mengunduh dan menyimpan Text Encoder...")
    text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL, subfolder="text_encoder")
    text_encoder.save_pretrained(os.path.join(LOCAL_DIR, "text_encoder"))
    print("[OK] Text Encoder berhasil disimpan.\n")
    
    # 3. Download & Save VAE
    print("3/3. Mengunduh dan menyimpan VAE...")
    vae = AutoencoderKL.from_pretrained(BASE_MODEL, subfolder="vae")
    vae.save_pretrained(os.path.join(LOCAL_DIR, "vae"))
    print("[OK] VAE berhasil disimpan.\n")
    
    print("=" * 60)
    print("SELESAI! Semua komponen model berhasil disimpan secara lokal.")
    print("Lokasi: " + LOCAL_DIR)
    print("=" * 60)

if __name__ == "__main__":
    main()
