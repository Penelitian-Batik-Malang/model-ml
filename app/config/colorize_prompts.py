# ============================================================
# KONFIGURASI PIPELINE: FINE-TUNED COLORIZER
# ============================================================
import os
from pathlib import Path

DEFAULT_STEPS = 20
DEFAULT_CFG_SCALE = 12.0
DEFAULT_COLOR_SCALE = 0.8
DEFAULT_SEED = 42

# 12 Template Prompts untuk Colorizer
COLORIZE_PROMPTS = {
    1:  ("Elegan Monokrom",
         "latar belakang hitam pekat, emas mawar, putih krem, batik monokrom elegan",
         "(solid pure black background:1.5), (rose gold:1.4), (creamy white:1.2), elegant monochrome batik",
         "blue, green, bright colors, neon, rainbow"),
    2:  ("Bumi Klasik",
         "latar belakang terakota dalam, tanah liat siena, cokelat tua, kuning oker, batik tanah klasik",
         "(solid deep terracotta background:1.5), (burnt sienna:1.3), (dark chocolate brown:1.4), (ochre yellow:1.2), classic earthy batik",
         "blue, purple, pink, bright neon, neon green, cyan"),
    3:  ("Laut Ketenangan",
         "latar belakang biru dongker dalam, biru safir, biru keabu-abuan, batik laut tenang",
         "(solid deep navy background:1.5), (sapphire blue:1.4), (dusty slate blue:1.2), tranquil sea batik",
         "red, bright orange, hot pink, green, neon"),
    4:  ("Hutan Tropis",
         "latar belakang hijau zamrud tua, hijau hutan, cokelat cokelat, batik hutan tropis",
         "(solid deep emerald green background:1.5), (forest green:1.4), (chocolate brown:1.2), tropical jungle batik",
         "red, purple, bright blue, orange, neon, pink"),
    5:  ("Sunset Pantai",
         "latar belakang oranye terbakar hangat, kuning amber, merah tua, batik pantai saat matahari terbenam",
         "(solid warm burnt orange background:1.5), (golden amber:1.4), (deep scarlet:1.2), sunset coastal batik",
         "blue, green, teal, purple, cold tones, grey"),
    6:  ("Laut Dalam",
         "latar belakang biru dongker dalam, teal samudra, pirus, batik laut dalam",
         "(solid deep navy blue background:1.6), (ocean teal:1.4), (turquoise:1.3), deep ocean batik",
         "red, orange, brown, warm colors, yellow, pink"),
    7:  ("Royal Purple",
         "latar belakang ungu violet kaya, ungu tua, emas metalik, batik mewah megah",
         "(solid rich violet purple background:1.5), (deep purple:1.4), (metallic gold:1.3), regal luxury batik",
         "blue, green, red, orange, teal, neon"),
    8:  ("Midnight Gold",
         "latar belakang hitam pekat, arang tua, emas metalik, batik tengah malam mewah",
         "(solid jet black background:1.5), (deep charcoal:1.3), (metallic gold:1.4), luxury midnight batik",
         "white, bright colors, pastel, blue, red"),
    9:  ("Pastel Garden",
         "latar belakang merah muda merona, lavender lembut, biru bubuk, batik taman pastel yang halus",
         "(solid blush pink background:1.4), (soft lavender:1.3), (powder blue:1.2), delicate pastel garden batik",
         "black, dark brown, deep red, neon, high contrast"),
    10: ("Sakura",
         "latar belakang merah muda bunga sakura, putih mutiara, hijau sage lembut, batik bunga sakura",
         "(solid cherry blossom pink background:1.5), (pearl white:1.3), (soft sage green:1.2), sakura blossom batik",
         "dark colors, black, deep blue, brown, red"),
    11: ("Harmoni Alam",
         "latar belakang hijau zaitun, hijau hutan tua, kuning mustar hangat, harmoni tanah analog",
         "(solid olive green background:1.5), (deep forest green:1.4), (warm mustard yellow:1.3), analogous earthy harmony",
         "red, purple, bright blue, pink, neon, cold tones"),
    12: ("Lembayung Sutra",
         "latar belakang violet dalam, ungu kecubung, hijau sage, emas amber, batik sutra triadik",
         "(solid deep violet background:1.7), (amethyst purple:1.5), (sage green:1.3), (amber gold:1.2), triadic silk batik",
         "blue, red, bright yellow, cyan, neon, black and white"),
}

DEFAULT_NEG_PROMPT = "grainy black-and-white photo, old box camera, grayscale photography, extra patterns in background, hallucinated details, messy background"

# Path model colorizer dipindahkan ke folder models/colorizer/
_CURRENT_DIR = Path(__file__).resolve().parent
COLORIZER_MODELS_DIR = _CURRENT_DIR.parents[1] / "models" / "colorizer"

UNET_MODEL_DIR = str(COLORIZER_MODELS_DIR / "unet")

local_base = COLORIZER_MODELS_DIR / "stable-diffusion-v1-5"
if local_base.exists() and local_base.is_dir():
    SD_BASE_MODEL = str(local_base)
else:
    SD_BASE_MODEL = os.getenv("BASE_MODEL", "runwayml/stable-diffusion-v1-5")
