import os
import gc
import time
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

from app.config.colorize_prompts import UNET_MODEL_DIR, SD_BASE_MODEL

class BatikColorizer:
    def __init__(self):
        print(f"torch {torch.__version__} | device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.t_max = 1000
        self._is_loaded = False
        
        self.gray_tf = transforms.Compose([
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def load(self):
        if self._is_loaded:
            return True

        try:
            print("\n[Colorizer] Memuat model Fine-Tuned dari:", UNET_MODEL_DIR)
            print("[Colorizer] Memuat Base Model dari:", SD_BASE_MODEL)
            self.tokenizer = CLIPTokenizer.from_pretrained(SD_BASE_MODEL, subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained(SD_BASE_MODEL, subfolder="text_encoder", torch_dtype=self.dtype).to(self.device)
            self.vae = AutoencoderKL.from_pretrained(SD_BASE_MODEL, subfolder="vae", torch_dtype=self.dtype).to(self.device)
            self.unet = UNet2DConditionModel.from_pretrained(UNET_MODEL_DIR, torch_dtype=self.dtype).to(self.device)
            
            self.vae.eval()
            self.text_encoder.eval()
            self.unet.eval()
            self._is_loaded = True
            print("[Colorizer] ✅ Model berhasil dimuat (LOCAL)")
            return True
        except Exception as e:
            print(f"[Colorizer] ❌ Gagal memuat model: {e}")
            return False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def unload(self):
        """Hapus semua model dari VRAM untuk membebaskan memory (on-demand switching)."""
        print("[Colorizer] Meng-unload model dari memory...")
        if hasattr(self, 'unet'): del self.unet
        if hasattr(self, 'vae'): del self.vae
        if hasattr(self, 'text_encoder'): del self.text_encoder
        if hasattr(self, 'tokenizer'): del self.tokenizer
        self.unet = None
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        self._is_loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[Colorizer] ✅ Model berhasil di-unload.")

    @torch.no_grad()
    def encode_img(self, t):
        return self.vae.encode(t.to(self.device, dtype=self.dtype)).latent_dist.sample() * self.vae.config.scaling_factor

    @torch.no_grad()
    def encode_txt(self, text):
        tok = self.tokenizer(text, padding="max_length",
                        max_length=self.tokenizer.model_max_length,
                        truncation=True, return_tensors="pt")
        return self.text_encoder(tok.input_ids.to(self.device))[0]

    def decode_lat(self, lat):
        dec = self.vae.decode(lat / self.vae.config.scaling_factor).sample
        dec = (dec / 2 + 0.5).clamp(0, 1)
        return Image.fromarray(
            (dec[0].permute(1,2,0).cpu().float().numpy()*255).astype(np.uint8))

    def replace_luma(self, col, gray):
        # Hard Luma Replacement (100%) - Menjamin SSIM/PSNR tinggi
        cl = np.array(col.convert("RGB").convert("LAB")).astype(np.float32)
        gl = np.array(gray.convert("RGB").convert("LAB")).astype(np.float32)
        cl[:,:,0] = gl[:,:,0]
        return Image.fromarray(cl.astype(np.uint8), "LAB").convert("RGB")

    def build_full_prompt(self, label, active_prompt):
        if not active_prompt:
            return ""
            
        # Logika Material Texture
        if any(x in label for x in ["Royal", "Midnight", "Emerald", "Peacock", "Purple"]):
            material = "premium silk satin texture, luxurious sheen, metallic thread accents"
        elif any(x in label for x in ["Sogan", "Classic", "Keraton"]):
            material = "fine primissima cotton texture, traditional wax-resist technique, matte finish"
        elif any(x in label for x in ["Pastel", "Sakura", "Mint", "Lavender"]):
            material = "soft linen fabric texture, airy and light textile feel"
        else:
            material = "high-quality batik textile texture, smooth fabric finish"

        full_prompt = f"{active_prompt}, {material}, clean solid background, masterpiece, best quality"
        return full_prompt

    @torch.no_grad()
    def colorize(self, img_pil, prompt, neg_prompt, steps, cfg, c_scale, seed=42):
        # Set seed for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        start_time = time.time()
        
        img_rgb  = img_pil.convert("RGB").resize((512, 512), Image.LANCZOS)
        img_gray = img_rgb.convert("L").convert("RGB")
        z_gray   = self.encode_img(self.gray_tf(img_rgb).unsqueeze(0))
        pos_emb  = self.encode_txt(prompt)
        neg_emb  = self.encode_txt(neg_prompt)
        
        t_sched  = list(range(0, self.t_max, self.t_max // steps))
        z_t = z_gray.clone()
        dt = 1.0 / len(t_sched)
        
        print(f"  [Colorizer] Menjalankan {len(t_sched)} steps (Rectified Flow Forward)...")
        for i, t_val in enumerate(t_sched):
            t_tensor = torch.tensor([t_val], dtype=torch.long, device=self.device)
            latent_model_input = torch.cat([z_t, z_gray], dim=1)
            
            pred_pos = self.unet(latent_model_input, t_tensor, encoder_hidden_states=pos_emb).sample
            pred_neg = self.unet(latent_model_input, t_tensor, encoder_hidden_states=neg_emb).sample
            v_pred = pred_neg + cfg * (pred_pos - pred_neg)
            z_t = z_t + v_pred * dt
                
        z_colored = z_gray + c_scale * (z_t - z_gray)
        ai_color_raw = self.decode_lat(z_colored)
        
        # High-res fusion
        ai_color_upscaled = ai_color_raw.resize(img_pil.size, Image.LANCZOS)
        final_img = self.replace_luma(ai_color_upscaled, img_pil)

        # ── SSIM & PSNR: RGB gambar asli (input) vs RGB hasil colorisasi (output) ──
        size = (512, 512)
        rgb_in  = np.array(img_pil.convert('RGB').resize(size))   # gambar asli
        rgb_out = np.array(final_img.convert('RGB').resize(size))  # hasil colorisasi
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

# Global instance untuk colorizer
colorizer_engine = BatikColorizer()
