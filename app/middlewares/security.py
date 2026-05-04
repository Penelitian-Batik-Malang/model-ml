import logging
import os
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware untuk menambahkan security headers ke semua response."""

    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Cache-Control": "no-store",
    }

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        for header_name, header_value in self.SECURITY_HEADERS.items():
            response.headers[header_name] = header_value
        return response


async def verify_api_key(request: Request, skip_paths: list = None) -> bool:
    """
    Validasi API Key dari header X-API-Key.
    Skip paths yang tidak memerlukan autentikasi (e.g., /api/health)
    
    Args:
        request: FastAPI Request object
        skip_paths: List path yang skip validation (default: ["/api/health"])
        
    Returns:
        True jika valid, raises HTTPException jika invalid
    """
    if skip_paths is None:
        skip_paths = ["/api/health", "/docs", "/openapi.json", "/redoc"]
    
    # Skip validation untuk health check dan dokumentasi
    if request.url.path in skip_paths:
        return True
    
    api_key = request.headers.get("X-API-Key")
    expected_api_key = os.getenv("API_KEY")
    
    if not api_key:
        logger.warning(f"Request tanpa API Key dari {request.client.host}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header"
        )
    
    if not expected_api_key:
        logger.error("API_KEY environment variable not set")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
    
    if api_key != expected_api_key:
        logger.warning(f"Invalid API Key attempt dari {request.client.host}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    
    return True


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware untuk validasi API Key di semua endpoint kecuali skip paths."""
    
    def __init__(self, app, skip_paths: list = None):
        super().__init__(app)
        self.skip_paths = skip_paths or ["/api/health", "/docs", "/openapi.json", "/redoc"]

    async def dispatch(self, request: Request, call_next):
        path = (request.scope.get("path") or request.url.path or "").rstrip("/") or "/"
        skip_paths = {item.rstrip("/") or "/" for item in self.skip_paths}

        if path in skip_paths or path.endswith("/health"):
            return await call_next(request)

        api_key = request.headers.get("X-API-Key")
        expected_api_key = os.getenv("API_KEY", "")

        if not api_key:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "status": 401,
                    "message": "Missing X-API-Key header",
                    "data": None,
                    "errors": ["Missing X-API-Key header"],
                    "meta": None,
                },
            )

        if not expected_api_key:
            logger.error("API_KEY environment variable not set")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "status": 500,
                    "message": "Internal server error",
                    "data": None,
                    "errors": ["Internal server error"],
                    "meta": None,
                },
            )

        if api_key != expected_api_key:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "status": 401,
                    "message": "Invalid API Key",
                    "data": None,
                    "errors": ["Invalid API Key"],
                    "meta": None,
                },
            )

        return await call_next(request)
