# pyrefly: ignore [missing-import]
from fastapi import FastAPI, File, UploadFile, Form
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from pydantic import BaseModel
import io
import base64
# pyrefly: ignore [missing-import]
from PIL import Image
# pyrefly: ignore [missing-import]
import uvicorn
from contextlib import asynccontextmanager
import asyncio

from config import (
    PROMPT_TEMPLATES,
    DEFAULT_STEPS, DEFAULT_CFG_SCALE, DEFAULT_COLOR_SCALE, DEFAULT_SEED, DEFAULT_NEG_PROMPT,
)
from inference_engine import BatikColorizer
# pyrefly: ignore [missing-import]
from deep_translator import GoogleTranslator

# ============================================================
# GLOBAL STATE
# ============================================================
translator = GoogleTranslator(source='auto', target='en')

finetuned_colorizer: BatikColorizer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global finetuned_colorizer

    print("Menginisialisasi Fine-Tuned pipeline...")
    finetuned_colorizer = BatikColorizer()
    print("✅ Fine-Tuned model siap.")
    yield
    print("Mematikan aplikasi...")


app = FastAPI(title="Batik AI Colorizer API", lifespan=lifespan)

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def image_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "finetuned_loaded": finetuned_colorizer is not None and finetuned_colorizer.is_loaded,
    }



# Global cache templates
cached_templates = None


@app.get("/api/templates")
def get_templates():
    global cached_templates

    if cached_templates:
        return {"templates": cached_templates}

    print("DEBUG: Menerjemahkan templates untuk pertama kali (Caching)...")
    templates = []
    to_indo = GoogleTranslator(source='auto', target='id')

    for t_id, (name, pos_indo, pos_eng, neg_eng) in PROMPT_TEMPLATES.items():
        try:
            neg_indo = to_indo.translate(neg_eng)
        except:
            neg_indo = neg_eng

        templates.append({
            "id": t_id,
            "name": name,
            "positive_indo": pos_indo,
            "positive_eng": pos_eng,
            "negative_eng": neg_eng,
            "negative_indo": neg_indo
        })

    cached_templates = templates
    return {"templates": templates}


@app.post("/api/colorize")
async def colorize_image(
    image: UploadFile = File(...),
    prompt_mode: str = Form("template"),
    template_id: int = Form(1),
    custom_prompt: str = Form(""),
    neg_prompt: str = Form(""),
    steps: int = Form(DEFAULT_STEPS),
    cfg_scale: float = Form(DEFAULT_CFG_SCALE),
    color_scale: float = Form(DEFAULT_COLOR_SCALE),
    seed: int = Form(DEFAULT_SEED),
):
    try:
        # ── Load image ──
        img_bytes = await image.read()
        img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        input_b64 = image_to_base64(img_pil)

        # ── Prepare prompts ──
        active_prompt = ""
        active_neg = neg_prompt if neg_prompt else DEFAULT_NEG_PROMPT
        label = ""
        prompt_indo = ""

        if prompt_mode == "template":
            if template_id in PROMPT_TEMPLATES:
                name, p_indo, p_eng, n_eng = PROMPT_TEMPLATES[template_id]
                active_prompt = p_eng
                active_neg = n_eng if not neg_prompt else neg_prompt
                label = name
                prompt_indo = p_indo
        elif prompt_mode == "custom":
            try:
                if custom_prompt.strip():
                    active_prompt = translator.translate(custom_prompt)
                    print(f"DEBUG: Translasi Positif '{custom_prompt}' -> '{active_prompt}'")
                else:
                    active_prompt = ""

                if neg_prompt.strip():
                    active_neg = translator.translate(neg_prompt)
                    print(f"DEBUG: Translasi Negatif '{neg_prompt}' -> '{active_neg}'")
                else:
                    active_neg = DEFAULT_NEG_PROMPT
            except Exception as e:
                print(f"ERROR: Translasi gagal: {e}")
                active_prompt = custom_prompt
                active_neg = neg_prompt if neg_prompt else DEFAULT_NEG_PROMPT

            label = "Custom"
            prompt_indo = custom_prompt

        # ── Pastikan pipeline sudah aktif ──
        print(f"[API] Mode: {prompt_mode} | File: {image.filename}")

        full_prompt = finetuned_colorizer.build_full_prompt(label, active_prompt)
        full_neg = f"{active_neg}, (extra patterns in background:1.3), (hallucinated details:1.3), (messy background:1.2), grainy black-and-white photo, grayscale photography"

        final_img, metrics = finetuned_colorizer.colorize(
            img_pil, full_prompt, full_neg,
            steps=steps, cfg=cfg_scale, c_scale=color_scale, seed=seed
        )

        metrics["pipeline"] = "finetuned"

        output_b64 = image_to_base64(final_img)

        return {
            "success": True,
            "input_image_b64": input_b64,
            "output_image_b64": output_b64,
            "metrics": metrics,
            "prompt_used": {
                "positive_indo": prompt_indo,
                "positive_eng": active_prompt,
                "full_prompt_eng": full_prompt,
                "negative": full_neg
            },
            "template_name": label,
            "pipeline_mode": "finetuned",
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
