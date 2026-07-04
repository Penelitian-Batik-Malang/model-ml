from typing import Any, List, Optional
from app.schemas.response import APIResponse, Meta


class ResponseBuilder:
    """Builder untuk membuat response standard API."""

    @staticmethod
    def success(
        data: Any = None,
        message: str = "Success",
        status: int = 200,
        meta: Optional[Meta] = None,
    ) -> APIResponse:
        """
        Membuat response sukses.
        
        Args:
            data: Data response (bisa list, dict, atau None)
            message: Pesan response
            status: HTTP status code
            meta: Metadata response (untuk pagination)
            
        Returns:
            APIResponse
        """
        return APIResponse(
            status=status,
            message=message,
            data=data,
            errors=[],
            meta=meta,
        )

    @staticmethod
    def error(
        message: str = "Error",
        status: int = 400,
        errors: Optional[List[str]] = None,
        data: Any = None,
    ) -> APIResponse:
        """
        Membuat response error.
        
        Args:
            message: Pesan error
            status: HTTP status code
            errors: List error messages
            data: Data tambahan (optional)
            
        Returns:
            APIResponse
        """
        if errors is None:
            errors = [message]
        
        return APIResponse(
            status=status,
            message=message,
            data=data,
            errors=errors,
            meta=None,
        )

    @staticmethod
    def paginated(
        data: List[Any],
        page: int = 1,
        page_size: int = 10,
        total_item: int = 0,
        message: str = "Success",
        status: int = 200,
    ) -> APIResponse:
        """
        Membuat response dengan pagination.
        
        Args:
            data: List data
            page: Halaman saat ini
            page_size: Jumlah item per halaman
            total_item: Total item seluruhnya
            message: Pesan response
            status: HTTP status code
            
        Returns:
            APIResponse
        """
        total_page = (total_item + page_size - 1) // page_size
        
        meta = Meta(
            size=len(data),
            page=page,
            total_page=total_page,
            total_item=total_item,
        )
        
        return APIResponse(
            status=status,
            message=message,
            data=data,
            errors=[],
            meta=meta,
        )


def to_dict(response: APIResponse) -> dict:
    """Convert APIResponse ke dict untuk serialization."""
    return {
        "status": response.status,
        "message": response.message,
        "data": response.data,
        "errors": response.errors,
        "meta": response.meta.model_dump() if response.meta is not None else None,
    }
