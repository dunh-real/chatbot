from fastapi import HTTPException, Header, Request
from typing import Optional

ALLOWED_TENANTS = {"viettel", "fpt", "vnpt", "vinfast"}  # Sau này load từ DB

def inject_tenant_id(
    request: Request = None,
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID")
) -> str:
    """
    Dependency để inject và validate tenant_id.
    Có thể gọi từ endpoint hoặc từ manager.
    """
    tenant_id = x_tenant_id or request.query_params.get("tenant_id") if request else None

    if not tenant_id or tenant_id not in ALLOWED_TENANTS:
        raise HTTPException(
            status_code=403,
            detail=f"Invalid or missing tenant_id. Allowed: {', '.join(ALLOWED_TENANTS)}"
        )

    return tenant_id