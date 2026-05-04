from slowapi import Limiter
from slowapi.util import get_remote_address
from app.config.settings import settings

# Create limiter instance (global limit applies to all routes)
limiter = Limiter(key_func=get_remote_address, default_limits=[settings.RATE_LIMIT_GLOBAL])

# Rate limit string constants
CLASSIFY_LIMIT = settings.RATE_LIMIT_CLASSIFY  # 10 requests per minute per IP
CBIR_LIMIT = settings.RATE_LIMIT_CBIR  # 5 requests per minute per IP
HEALTH_LIMIT = settings.RATE_LIMIT_HEALTH  # 60 requests per minute per IP
GLOBAL_LIMIT = settings.RATE_LIMIT_GLOBAL  # global cap

# Daftar rate limit config
RATE_LIMITS = {
    "classify": {
        "limit": CLASSIFY_LIMIT,
        "description": "10 requests per minute per IP (inference heavy)"
    },
    "cbir": {
        "limit": CBIR_LIMIT,
        "description": "5 requests per minute per IP"
    },
    "health": {
        "limit": HEALTH_LIMIT,
        "description": "60 requests per minute per IP"
    },
}
