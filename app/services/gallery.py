import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from app.config.settings import settings

NPZ_PATH = settings.NPZ_PATH
CSV_PATH = settings.CSV_PATH
IMAGE_ROOT = settings.IMAGE_ROOT
IMG_SIZE = settings.IMG_SIZE
BATCH_SIZE = settings.BATCH_SIZE

infer_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@torch.no_grad()
def _encode_images_from_df(model, dataframe, transform, device, image_root, batch_size=32):
    embs, paths, cats = [], [], []
    rows = dataframe.reset_index(drop=True)

    for i in range(0, len(rows), batch_size):
        chunk = rows.iloc[i:i + batch_size]
        imgs = []
        for _, r in chunk.iterrows():
            p = os.path.join(image_root, r["folder_category"], r["file_name"])
            imgs.append(transform(Image.open(p).convert("RGB")))
            paths.append(p)
            cats.append(r["folder_category"])

        x = torch.stack(imgs).to(device)
        e = F.normalize(model.image_projection(model.image_encoder(x)), p=2, dim=1)
        embs.append(e.cpu())

    return torch.cat(embs, 0), paths, np.array(cats)

def build_gallery_if_missing(model, device):
    if os.path.exists(NPZ_PATH):
        print(f"[Startup] Loading gallery: {NPZ_PATH}")
        npz = np.load(NPZ_PATH, allow_pickle=True)

        gallery_embeddings = torch.tensor(npz["embeddings"], dtype=torch.float32)
        gallery_categories = npz["categories"].tolist()
        gallery_paths = [
            os.path.join(IMAGE_ROOT, str(cat), os.path.basename(str(p).replace("\\", "/")))
            for p, cat in zip(npz["paths"], npz["categories"])
        ]

        print(
            f"[Startup] Gallery OK     | "
            f"{len(gallery_paths)} gambar  "
            f"shape {gallery_embeddings.shape}"
        )
        return gallery_embeddings, gallery_paths, gallery_categories

    # ── .npz belum ada -> build dari CSV ──────────────────────────────────
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Gallery NPZ tidak ditemukan ({NPZ_PATH}) dan CSV_PATH juga tidak ada ({CSV_PATH}). "
            f"Tidak bisa membangun gallery embedding."
        )

    print(f"[Startup] {NPZ_PATH} tidak ditemukan. Membangun gallery dari CSV: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH, sep=';')
    df = df[
        (df["caption_1"].astype(str).str.strip().str.lower() != "gagal deskripsi otomatis")
        & (df["caption_2"].astype(str).str.strip().str.lower() != "gagal deskripsi otomatis")
    ].reset_index(drop=True)

    print(f"[Startup] Jumlah data (CSV): {len(df)} | kategori: {df['folder_category'].nunique()}")

    gallery_emb_tensor, gallery_paths, gallery_categories_arr = _encode_images_from_df(
        model, df, infer_transform, device, IMAGE_ROOT, batch_size=BATCH_SIZE
    )
    gallery_categories = gallery_categories_arr.tolist()

    os.makedirs(os.path.dirname(NPZ_PATH) or ".", exist_ok=True)
    np.savez(
        NPZ_PATH,
        embeddings=gallery_emb_tensor.numpy(),
        categories=np.array(gallery_categories, dtype=object),
        paths=np.array(gallery_paths, dtype=object),
    )

    print(f"[Startup] Gallery baru dibangun & disimpan ke: {NPZ_PATH}")
    print(f"[Startup] Shape: {gallery_emb_tensor.shape} | kategori: {len(set(gallery_categories))}")

    return gallery_emb_tensor, gallery_paths, gallery_categories
