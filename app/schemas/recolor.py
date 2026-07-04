from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator
import re


_HEX_PATTERN = re.compile(r'^#[0-9A-Fa-f]{6}$')


class PaletteExtractRequest(BaseModel):
    method: str = Field(default="all", description="kmeans | histogram | median_cut | all")
    n_colors: int = Field(default=6, ge=1, le=20, description="Number of palette colors (1-20)")

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str):
        allowed = {"kmeans", "histogram", "median_cut", "all"}
        if v.lower() not in allowed:
            raise ValueError(f"method must be one of {allowed}")
        return v.lower()


class RecolorRequest(BaseModel):
    palette: List[str] = Field(..., min_length=1, max_length=10, description="List of hex colors")
    white_threshold: float = Field(default=150.0, ge=0, le=765, description="White threshold 0-765")

    @field_validator("palette")
    @classmethod
    def validate_palette(cls, v: List[str]):
        for color in v:
            if not _HEX_PATTERN.match(color):
                raise ValueError(f"Invalid hex color: {color}. Must be #RRGGBB format.")
        return v


class RecolorSimpleRequest(BaseModel):
    palette: List[str] = Field(..., min_length=1, max_length=10, description="List of hex colors")

    @field_validator("palette")
    @classmethod
    def validate_palette(cls, v: List[str]):
        for color in v:
            if not _HEX_PATTERN.match(color):
                raise ValueError(f"Invalid hex color: {color}. Must be #RRGGBB format.")
        return v
