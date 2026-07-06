import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import timm

class ImageEncoder(nn.Module):
    def __init__(self, name: str = "convnext_small"):
        super().__init__()
        self.backbone   = timm.create_model(name, pretrained=False, num_classes=0)
        self.output_dim = self.backbone.num_features
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class TextEncoder(nn.Module):
    def __init__(self, name: str = "indobenchmark/indobert-base-p1"):
        super().__init__()
        self.bert       = AutoModel.from_pretrained(name)
        self.output_dim = self.bert.config.hidden_size
 
    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.bert(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0, :]
 
 
class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        self.ln = nn.LayerNorm(out_dim)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(self.net(x))
 
 
class MultimodalRetrievalModel(nn.Module):
    def __init__(self, proj_dim: int = 768):
        super().__init__()
        self.image_encoder    = ImageEncoder("convnext_small")
        self.text_encoder     = TextEncoder("indobenchmark/indobert-base-p1")
        self.image_projection = ProjectionHead(self.image_encoder.output_dim, proj_dim)
        self.text_projection  = ProjectionHead(self.text_encoder.output_dim,  proj_dim)
        self.logit_scale      = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
 
    # ── Shortcut helpers untuk inference ─────────────────────────────────
 
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(
            self.image_projection(self.image_encoder(images)), p=2, dim=1
        )
 
    def encode_text(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return F.normalize(
            self.text_projection(self.text_encoder(ids, mask)), p=2, dim=1
        )
 
    def forward(
        self,
        images: torch.Tensor,
        ids: torch.Tensor,
        mask: torch.Tensor,
    ):
        return self.encode_image(images), self.encode_text(ids, mask)
