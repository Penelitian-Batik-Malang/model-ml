# Required Model Files

The following files are required in this directory for the ML API to function correctly. They are not included in the repository due to their size.

| Filename | Description | Size |
|----------|-------------|------|
| `augmentTest_batik_cnn_pararel_elu3.h5` | Batik CNN model (parallel elu) | ~11.8 MB |
| `features_768_features.npy` | Feature vectors for batik search | ~46.9 MB |
| `features_768_indexed_database.csv` | Indexed database for features | ~3.2 MB |
| `features_768_kmeans_model.pkl` | K-Means model for feature clustering | ~68 KB |
| `label_mapping_pararelEluAugment3.json` | Label mapping for batik classification | ~1 KB |
| `label_mapping_pararelEluAugment13.json` | Label mapping for batik classification (v13) | ~1 KB |
| `model_ConvNextTiny_original_all.pt` | ConvNext Tiny model weights | ~112.9 MB |
| `models_pararelElu_augment_13.h5` | Final batik model weights | ~11.8 MB |

Please ensure these files are placed in the `models/` directory before running the API.
