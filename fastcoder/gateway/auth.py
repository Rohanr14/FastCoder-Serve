"""Bearer-token authentication for the gateway."""

from __future__ import annotations

from fastapi import HTTPException, Request, status

from fastcoder.gateway.config import GatewaySettings


def extract_bearer_token(authorization_header: str | None) -> str | None:
    """Extract a bearer token from an Authorization header."""

    if not authorization_header:
        return None
    scheme, _, token = authorization_header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token.strip()


async def require_bearer_token(request: Request, settings: GatewaySettings) -> str:
    """Validate bearer auth unless explicit test/local bypass is enabled."""

    token = extract_bearer_token(request.headers.get("authorization"))
    if settings.auth_bypass:
        return token or "auth-bypass"
    if token != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token
