# ============================================================
# KONFIGURASI BACKEND COLORIZER
# ============================================================
import os
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

# Load dari file .env
load_dotenv()

# Path folder unet yang sudah didownload
MODEL_DIR  = os.getenv("MODEL_DIR", "D:/Recolor/backend/unet")

# Base model Stable Diffusion
BASE_MODEL = os.getenv("BASE_MODEL", "runwayml/stable-diffusion-v1-5")

# ============================================================
# KONFIGURASI PIPELINE: FINE-TUNED (existing)
# ============================================================
DEFAULT_STEPS = 50
DEFAULT_CFG_SCALE = 12.0
DEFAULT_COLOR_SCALE = 0.8
DEFAULT_SEED = 42


# ── DAFTAR TEMPLATE PROMPT (ID, Nama, Positif Indo, Positive Eng, Negative Eng) ──────────
PROMPT_TEMPLATES = {
    1:  ("Sogan Indigo",   
         "latar belakang indigo tua pekat, biru indigo dalam, cokelat sienna hangat, gading antik, batik tradisional",
         "(solid dark indigo background:1.5), (deep indigo blue:1.4), (warm sienna brown:1.3), (antique ivory:1.2), traditional batik",
         "red, orange, bright yellow, neon, green, bright pink"),
    2:  ("Batik Pesisir",  
         "latar belakang merah tua terang, teal cerah, kuning keemasan, batik pesisir yang hidup",
         "(solid crimson red background:1.5), (bright teal:1.4), (golden yellow:1.3), vibrant coastal batik",
         "dark brown, black, purple, dull colors, monochrome, grey"),
    3:  ("Batik Keraton",  
         "latar belakang indigo dalam, krem gading antik, emas metalik, batik istana kerajaan",
         "(solid deep indigo background:1.5), (antique ivory cream:1.3), (metallic gold:1.3), royal court batik",
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

