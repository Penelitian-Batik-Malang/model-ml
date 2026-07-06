import argparse
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Lihat isi file .npz gallery embedding")
    parser.add_argument("npz_path", help="Path ke file .npz, contoh: gallery_v3.npz")
    parser.add_argument("--show", type=int, default=5, help="Jumlah baris contoh yang ditampilkan (default: 5)")
    parser.add_argument("--category", type=str, default=None, help="Filter: tampilkan hanya baris dari kategori ini")
    args = parser.parse_args()

    try:
        data = np.load(args.npz_path, allow_pickle=True)
    except FileNotFoundError:
        print(f"File tidak ditemukan: {args.npz_path}")
        sys.exit(1)

    print("=" * 60)
    print(f"FILE       : {args.npz_path}")
    print(f"KEYS       : {list(data.files)}")
    print("=" * 60)

    # ── Info tiap array di dalam .npz ────────────────────────────────────
    for key in data.files:
        arr = data[key]
        print(f"\n[{key}]")
        print(f"  shape : {arr.shape}")
        print(f"  dtype : {arr.dtype}")

    embeddings = data["embeddings"] if "embeddings" in data.files else None
    paths      = data["paths"].tolist() if "paths" in data.files else None
    categories = data["categories"].tolist() if "categories" in data.files else None

    # ── Ringkasan embedding ───────────────────────────────────────────────
    if embeddings is not None:
        norms = np.linalg.norm(embeddings, axis=1)
        print("\n" + "=" * 60)
        print("RINGKASAN EMBEDDING")
        print("=" * 60)
        print(f"Total vektor       : {embeddings.shape[0]}")
        print(f"Dimensi embedding  : {embeddings.shape[1]}")
        print(f"Rata-rata L2 norm  : {norms.mean():.4f}  (idealnya ≈ 1.0 kalau sudah di-normalize)")
        print(f"Min / Max L2 norm  : {norms.min():.4f} / {norms.max():.4f}")
        if not np.allclose(norms, 1.0, atol=1e-2):
            print("  PERINGATAN: sebagian besar norm TIDAK ≈ 1.0 — embedding kemungkinan belum di-L2-normalize!")

    # ── Ringkasan kategori ────────────────────────────────────────────────
    if categories is not None:
        unique, counts = np.unique(categories, return_counts=True)
        print("\n" + "=" * 60)
        print(f"RINGKASAN KATEGORI ({len(unique)} kategori)")
        print("=" * 60)
        for cat, count in sorted(zip(unique, counts), key=lambda x: -x[1]):
            print(f"  {cat:<40s} : {count} gambar")

    # ── Filter kategori tertentu (opsional) ────────────────────────────────
    if args.category and categories is not None:
        print("\n" + "=" * 60)
        print(f"BARIS DENGAN KATEGORI = '{args.category}'")
        print("=" * 60)
        matched = [i for i, c in enumerate(categories) if c == args.category]
        if not matched:
            print("  Tidak ada baris yang cocok.")
        for i in matched[:args.show]:
            path_info = paths[i] if paths else "-"
            print(f"  idx={i:<6d} category={categories[i]:<25s} path={path_info}")
        if len(matched) > args.show:
            print(f"  ... dan {len(matched) - args.show} baris lainnya")
        return

    # ── Contoh N baris pertama ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"CONTOH {args.show} BARIS PERTAMA")
    print("=" * 60)
    n = embeddings.shape[0] if embeddings is not None else (len(paths) if paths else 0)
    for i in range(min(args.show, n)):
        cat = categories[i] if categories else "-"
        path_info = paths[i] if paths else "-"
        emb_preview = np.array2string(embeddings[i][:5], precision=4) if embeddings is not None else "-"
        print(f"  idx={i:<6d} category={cat:<25s} path={path_info}")
        print(f"           embedding[:5] = {emb_preview} ...")


if __name__ == "__main__":
    main()
