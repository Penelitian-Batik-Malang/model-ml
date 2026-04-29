import os
import sys
import numpy as np
import numpy.core as _np_core
import pandas as pd
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import torch
from torchvision import models, transforms

from config import BATIK_SEARCH_FEATURES_NPY, BATIK_SEARCH_KMEANS_MODEL, BATIK_SEARCH_INDEXED_DB_CSV

# Shim: KMeans pkl disimpan dengan numpy>=1.25 yang punya numpy._core,
# tapi numpy<1.25 hanya punya numpy.core. Map supaya pickle bisa resolve.
sys.modules.setdefault("numpy._core", _np_core)
sys.modules.setdefault("numpy._core.multiarray", _np_core.multiarray)
sys.modules.setdefault("numpy._core.numeric", _np_core.numeric)
sys.modules.setdefault("numpy._core.umath", _np_core.umath)

DEFAULT_FEATURES_PATH = str(BATIK_SEARCH_FEATURES_NPY)
DEFAULT_KMEANS_PATH = str(BATIK_SEARCH_KMEANS_MODEL)
DEFAULT_INDEXED_DB_PATH = str(BATIK_SEARCH_INDEXED_DB_CSV)

features = np.array([])
kmeans = None
cluster_df = pd.DataFrame()
feature_extractor = None
FEATURE_EXTRACTOR_WEIGHTS = os.getenv("BATIK_SEARCH_FEATURE_EXTRACTOR_WEIGHTS", "").strip()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

query_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_cbir_models():
    global features, kmeans, cluster_df, feature_extractor
    
    try:
        if os.path.exists(DEFAULT_FEATURES_PATH):
            features = np.load(DEFAULT_FEATURES_PATH)
        else:
            print(f"[Batik Search] No NPY features file: {DEFAULT_FEATURES_PATH}")

        if os.path.exists(DEFAULT_KMEANS_PATH):
            kmeans = joblib.load(DEFAULT_KMEANS_PATH)
        else:
            print(f"[Batik Search] No KMeans model: {DEFAULT_KMEANS_PATH}")

        if os.path.exists(DEFAULT_INDEXED_DB_PATH):
            cluster_df = pd.read_csv(DEFAULT_INDEXED_DB_PATH)
        else:
            print("[Batik Search] Indexed database not found")
            
        feature_extractor = models.convnext_small(weights=None)
        feature_extractor.classifier[-1] = torch.nn.Identity()

        if FEATURE_EXTRACTOR_WEIGHTS and os.path.exists(FEATURE_EXTRACTOR_WEIGHTS):
            checkpoint = torch.load(FEATURE_EXTRACTOR_WEIGHTS, map_location=DEVICE)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            feature_extractor.load_state_dict(checkpoint)

        feature_extractor = feature_extractor.to(DEVICE).eval()
        
        print("[Batik Search] Models loaded successfully")
        return True
    except Exception as e:
        print(f"[Batik Search] Error loading models: {str(e)}")
        return False

def extract_feature(image_pil: Image.Image):
    if feature_extractor is None:
        load_cbir_models()
    
    img = image_pil.convert("RGB")
    tensor = query_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = feature_extractor(tensor).cpu().numpy()
    return feat

def retrieve_top_n(query_feature, top_n=10):
    if kmeans is None or len(cluster_df) == 0:
        load_cbir_models()
        
    if len(query_feature.shape) == 1:
        query_feature = query_feature.reshape(1, -1)
    
    cluster_id = int(kmeans.predict(query_feature)[0])
    cluster_members = cluster_df[cluster_df["cluster"] == cluster_id]
    
    if cluster_members.empty:
        return cluster_id, pd.DataFrame()
    
    member_indices = cluster_members.index
    member_features = features[member_indices]
    
    similarities = cosine_similarity(query_feature, member_features).flatten()
    
    result_df = cluster_members.copy()
    result_df["similarity"] = similarities
    result_df = result_df.sort_values(by="similarity", ascending=False).head(top_n)
    
    return cluster_id, result_df

def search_general_batik(image_pil: Image.Image, top_n=10) -> dict:
    try:
        query_feat = extract_feature(image_pil)
        cluster_id, top_images_df = retrieve_top_n(query_feat, top_n=top_n)
        
        if top_images_df.empty:
            return {"success": False, "message": "No similar images found", "cluster_id": cluster_id}
            
        results = []
        for _, row in top_images_df.iterrows():
            s3_path = str(row['path_s3']).replace('\\', '/')
            
            results.append({
                "path_s3": s3_path,
                "label": row.get('label', ''),
                "cluster": int(row['cluster']),
                "similarity": float(row['similarity']),
            })
            
        return {
            "success": True,
            "cluster_id": cluster_id,
            "results": results,
            "message": f"Found {len(results)} similar images in cluster {cluster_id}"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
