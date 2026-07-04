import io
import base64
import traceback
from PIL import Image

from fastapi import APIRouter, File, UploadFile, Form
from deep_translator import GoogleTranslator

from app.config.colorize_prompts import (
    COLORIZE_PROMPTS,
    DEFAULT_STEPS, 
    DEFAULT_CFG_SCALE, 
    DEFAULT_COLOR_SCALE, 
    DEFAULT_SEED, 
    DEFAULT_NEG_PROMPT,
)
from app.services.colorizer_engine import colorizer_engine

router = APIRouter(prefix="/colorizer")
translator = GoogleTranslator(source='auto', target='en')
cached_templates = None

def image_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


@router.get("/templates")
def get_templates():
    global cached_templates

    if cached_templates:
        return {"templates": cached_templates}

    print("DEBUG: Menerjemahkan templates untuk pertama kali (Caching)...")
    templates = []
    to_indo = GoogleTranslator(source='auto', target='id')

    for t_id, (name, pos_indo, pos_eng, neg_eng) in COLORIZE_PROMPTS.items():
        try:
            neg_indo = to_indo.translate(neg_eng)
        except Exception:
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


@router.post("/colorize")
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
        # Pengecekan status model
        if not colorizer_engine.is_loaded:
            return {"success": False, "error": "Model Colorizer belum siap atau gagal dimuat."}

        # ── Load image ──
        img_bytes = await image.read()
        img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # ── Prepare prompts ──
        active_prompt = ""
        active_neg = neg_prompt if neg_prompt else DEFAULT_NEG_PROMPT
        label = ""
        prompt_indo = ""

        if prompt_mode == "template":
            if template_id in COLORIZE_PROMPTS:
                name, p_indo, p_eng, n_eng = COLORIZE_PROMPTS[template_id]
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

        full_prompt = colorizer_engine.build_full_prompt(label, active_prompt)
        full_neg = f"{active_neg}, (extra patterns in background:1.3), (hallucinated details:1.3), (messy background:1.2), grainy black-and-white photo, grayscale photography"

        final_img, metrics = colorizer_engine.colorize(
            img_pil, full_prompt, full_neg,
            steps=steps, cfg=cfg_scale, c_scale=color_scale, seed=seed
        )

        metrics["pipeline"] = "finetuned"

        output_b64 = image_to_base64(final_img)

        return {
            "success": True,
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
        traceback.print_exc()
        return {"success": False, "error": str(e)}
