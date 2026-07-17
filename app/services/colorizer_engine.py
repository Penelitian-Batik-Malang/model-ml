import os
import gc
import time
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torchvision import transforms
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

from app.config.colorize_prompts import UNET_MODEL_DIR, SD_BASE_MODEL


def _resolve_dtype() -> torch.dtype:
    """Pilih dtype terbaik untuk CPU.
    
    - CUDA  → float16 (VRAM efisien)
    - CPU   → bfloat16 jika CPU modern (AVX-512 BFloat16), else float32
    bfloat16 setara float32 secara numerik tapi 2× lebih hemat RAM.
    """
    if torch.cuda.is_available():
        return torch.float16
    # bfloat16 didukung PyTorch CPU secara native (torch >= 1.11)
    # Operasi bfloat16 di CPU lebih lambat tapi lebih hemat RAM.
    # Untuk inferensi kita tetap float32 agar akurat, optimasi RAM via INT8.
    return torch.float32


class BatikColorizer:
    def __init__(self):
        print(f"torch {torch.__version__} | device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = _resolve_dtype()
        self.t_max = 1000
        self._is_loaded = False
        self._emb_cache: dict = {}

        # Set CPU thread count dari env (OMP_NUM_THREADS di HF Spaces)
        cpu_count = int(os.getenv("OMP_NUM_THREADS", str(os.cpu_count() or 2)))
        torch.set_num_threads(cpu_count)
        torch.set_num_interop_threads(max(1, cpu_count // 2))

        self.gray_tf = transforms.Compose([
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def load(self):
        if self._is_loaded:
            return True

        try:
            print("\n[Colorizer] Memuat model Fine-Tuned dari:", UNET_MODEL_DIR)
            print("[Colorizer] Memuat Base Model dari:", SD_BASE_MODEL)

            # ── Tokenizer & Text Encoder ──
            self.tokenizer = CLIPTokenizer.from_pretrained(SD_BASE_MODEL, subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained(
                SD_BASE_MODEL, subfolder="text_encoder", torch_dtype=self.dtype
            ).to(self.device)

            # ── VAE ──
            self.vae = AutoencoderKL.from_pretrained(
                SD_BASE_MODEL, subfolder="vae", torch_dtype=self.dtype
            ).to(self.device)

            # ── UNet: coba load INT8 quantized dulu, fallback ke FP32 ──
            int8_path = Path(UNET_MODEL_DIR) / "unet_int8_state_dict.pt"
            if int8_path.exists():
                print(f"[Colorizer] Memuat UNet INT8 dari: {int8_path}")
                import torch.ao.quantization as tq
                self.unet = UNet2DConditionModel.from_pretrained(
                    UNET_MODEL_DIR, torch_dtype=torch.float32
                )
                self.unet = tq.quantize_dynamic(
                    self.unet, {torch.nn.Linear}, dtype=torch.qint8
                )
                state = torch.load(str(int8_path), map_location="cpu", weights_only=True)
                self.unet.load_state_dict(state)
                print("[Colorizer] ✅ UNet INT8 loaded (~1.6 GB RAM)")
            else:
                print(f"[Colorizer] ⚠️  unet_int8_state_dict.pt tidak ditemukan, fallback ke FP32 (~6.4 GB RAM)")
                print(f"[Colorizer]    Jalankan: python scripts/quantize_unet.py untuk menghemat ~4.8 GB RAM")
                self.unet = UNet2DConditionModel.from_pretrained(
                    UNET_MODEL_DIR, torch_dtype=self.dtype
                ).to(self.device)

            self.vae.eval()
            self.text_encoder.eval()
            self.unet.eval()
            self._is_loaded = True
            print("[Colorizer] ✅ Model berhasil dimuat (LOCAL)")

            # ── Pre-cache semua 12 prompt embeddings (CLIP forward pass) ──
            # Karena prompt hanya 12 template yang fixed, encode sekali saja
            # dan cache hasilnya. Hemat ~200ms per request.
            self._build_prompt_cache()

            return True
        except Exception as e:
            print(f"[Colorizer] ❌ Gagal memuat model: {e}")
            return False

    def _build_prompt_cache(self):
        """Pre-encode semua 12 template prompt dan default neg prompt ke embeddings.
        
        Dipanggil sekali setelah model load. Menghemat 2× CLIP forward pass per request.
        """
        try:
            from app.config.colorize_prompts import COLORIZE_PROMPTS, DEFAULT_NEG_PROMPT
            self._emb_cache = {}
            with torch.no_grad():
                # Negative prompt default
                self._emb_cache["__neg__"] = self.encode_txt(DEFAULT_NEG_PROMPT).cpu()
                # Semua 12 template positive prompts
                for pid, (_, _, prompt_en, neg_en) in COLORIZE_PROMPTS.items():
                    self._emb_cache[f"pos_{pid}"] = self.encode_txt(prompt_en).cpu()
                    self._emb_cache[f"neg_{pid}"] = self.encode_txt(neg_en).cpu()
            print(f"[Colorizer] ✅ {len(self._emb_cache)} prompt embeddings di-cache (hemat ~200ms/request)")
        except Exception as e:
            print(f"[Colorizer] ⚠️  Gagal build prompt cache: {e}")
            self._emb_cache = {}

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def unload(self):
        """Hapus semua model dari memory untuk membebaskan RAM (on-demand switching)."""
        print("[Colorizer] Meng-unload model dari memory...")
        for attr in ("unet", "vae", "text_encoder", "tokenizer"):
            if hasattr(self, attr):
                delattr(self, attr)
        self.unet = None
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        self._is_loaded = False
        self._emb_cache = {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[Colorizer] ✅ Model berhasil di-unload.")

    @torch.no_grad()
    def encode_img(self, t):
        return self.vae.encode(t.to(self.device, dtype=self.dtype)).latent_dist.sample() * self.vae.config.scaling_factor

    @torch.no_grad()
    def encode_txt(self, text: str) -> torch.Tensor:
        tok = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return self.text_encoder(tok.input_ids.to(self.device))[0]

    def _get_emb(self, key: str, text: str) -> torch.Tensor:
        """Ambil embedding dari cache jika ada, fallback ke encode langsung."""
        if key in self._emb_cache:
            return self._emb_cache[key].to(self.device)
        return self.encode_txt(text)

    def decode_lat(self, lat):
        dec = self.vae.decode(lat / self.vae.config.scaling_factor).sample
        dec = (dec / 2 + 0.5).clamp(0, 1)
        return Image.fromarray(
            (dec[0].permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
        )

    def replace_luma(self, col, gray):
        """Hard Luma Replacement (100%) — Menjamin SSIM/PSNR tinggi."""
        cl = np.array(col.convert("RGB").convert("LAB")).astype(np.float32)
        gl = np.array(gray.convert("RGB").convert("LAB")).astype(np.float32)
        cl[:, :, 0] = gl[:, :, 0]
        return Image.fromarray(cl.astype(np.uint8), "LAB").convert("RGB")

    def build_full_prompt(self, label, active_prompt):
        if not active_prompt:
            return ""

        # Logika Material Texture berdasarkan label warna
        if any(x in label for x in ["Royal", "Midnight", "Emerald", "Peacock", "Purple"]):
            material = "premium silk satin texture, luxurious sheen, metallic thread accents"
        elif any(x in label for x in ["Sogan", "Classic", "Keraton"]):
            material = "fine primissima cotton texture, traditional wax-resist technique, matte finish"
        elif any(x in label for x in ["Pastel", "Sakura", "Mint", "Lavender"]):
            material = "soft linen fabric texture, airy and light textile feel"
        else:
            material = "high-quality batik textile texture, smooth fabric finish"

        return f"{active_prompt}, {material}, clean solid background, masterpiece, best quality"

    @torch.no_grad()
    def colorize(
        self,
        img_pil,
        prompt: str,
        neg_prompt: str,
        steps: int,
        cfg: float,
        c_scale: float,
        seed: int = 42,
        # Parameter opsional untuk cache hit (dari controller)
        template_id: int = None,
    ):
        # Set seed for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        start_time = time.time()

        img_rgb = img_pil.convert("RGB").resize((512, 512), Image.LANCZOS)
        img_gray = img_rgb.convert("L").convert("RGB")
        z_gray = self.encode_img(self.gray_tf(img_rgb).unsqueeze(0))

        # ── Gunakan cache embedding jika template_id tersedia ──
        if template_id is not None and f"pos_{template_id}" in self._emb_cache:
            pos_emb = self._emb_cache[f"pos_{template_id}"].to(self.device)
            neg_emb = self._emb_cache[f"neg_{template_id}"].to(self.device)
            print(f"  [Colorizer] ✅ Cache hit untuk template_id={template_id}")
        else:
            pos_emb = self.encode_txt(prompt)
            neg_emb = self.encode_txt(neg_prompt)

        t_sched = list(range(0, self.t_max, self.t_max // steps))
        z_t = z_gray.clone()
        dt = 1.0 / len(t_sched)

        print(f"  [Colorizer] Menjalankan {len(t_sched)} steps (Rectified Flow - BATCHED)...")
        # Batched embeddings: [neg_emb, pos_emb] → shape [2, 77, 768]
        encoder_hidden_states_batched = torch.cat([neg_emb, pos_emb], dim=0)

        for i, t_val in enumerate(t_sched):
            t_tensor = torch.tensor([t_val, t_val], dtype=torch.long, device=self.device)
            latent_model_input = torch.cat([z_t, z_gray], dim=1)
            # Batch input ke UNet: double batch dimension dari 1 ke 2
            latent_model_input_batched = torch.cat([latent_model_input, latent_model_input], dim=0)

            # Single forward pass untuk pos dan neg prompt
            pred_all = self.unet(
                latent_model_input_batched,
                t_tensor,
                encoder_hidden_states=encoder_hidden_states_batched,
            ).sample
            pred_neg, pred_pos = pred_all.chunk(2, dim=0)

            v_pred = pred_neg + cfg * (pred_pos - pred_neg)
            z_t = z_t + v_pred * dt

        z_colored = z_gray + c_scale * (z_t - z_gray)
        ai_color_raw = self.decode_lat(z_colored)

        # High-res fusion
        ai_color_upscaled = ai_color_raw.resize(img_pil.size, Image.LANCZOS)
        final_img = self.replace_luma(ai_color_upscaled, img_pil)

        # ── SSIM & PSNR ──
        size = (512, 512)
        rgb_in = np.array(img_pil.convert("RGB").resize(size))
        rgb_out = np.array(final_img.convert("RGB").resize(size))
        try:
            ssim_score = float(ssim_metric(rgb_in, rgb_out, data_range=255, channel_axis=2))
        except TypeError:
            ssim_score = float(ssim_metric(rgb_in, rgb_out, data_range=255, multichannel=True))
        psnr_score = float(psnr_metric(rgb_in, rgb_out, data_range=255))

        elapsed = time.time() - start_time

        metrics = {
            "time": float(elapsed),
            "ssim": round(ssim_score, 4),
            "psnr": round(psnr_score, 2),
        }

        return final_img, metrics


# Global instance untuk colorizer (dimuat lazy, bukan saat import)
colorizer_engine = BatikColorizer()
