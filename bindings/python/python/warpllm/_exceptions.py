from __future__ import annotations

import json
from typing import NoReturn


class WarpLLMError(Exception):
    """Base for all warpllm errors. `code` is a stable machine-readable slug."""

    def __init__(self, message: str, *, code: str = "error") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class InvalidRequestError(WarpLLMError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="invalid_request")


class APIConnectionError(WarpLLMError):
    def __init__(self, message: str, *, provider: str | None = None) -> None:
        super().__init__(message, code="connection_error")
        self.provider = provider


class APIStatusError(WarpLLMError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        provider: str | None = None,
        error_type: str | None = None,
        code: str = "provider_error",
    ) -> None:
        super().__init__(message, code=code)
        self.status_code = status_code
        self.provider = provider
        self.error_type = error_type


class AuthenticationError(APIStatusError):
    pass


class RateLimitError(APIStatusError):
    pass


def raise_from_wire(raw: str) -> NoReturn:
    """Translates the native layer's wire-format JSON into typed exceptions."""
    try:
        data = json.loads(raw)
    except ValueError:
        raise WarpLLMError(raw) from None

    code = data.get("code", "error")
    message = data.get("message", raw)
    provider = data.get("provider")

    if code == "not_implemented":
        raise NotImplementedError(message)
    if code == "invalid_request":
        raise InvalidRequestError(message)
    if code == "missing_api_key":
        raise AuthenticationError(message, provider=provider, code=code)
    if code == "connection_error":
        raise APIConnectionError(message, provider=provider)
    if code == "provider_error":
        status = data.get("status")
        kwargs = {
            "status_code": status,
            "provider": provider,
            "error_type": data.get("error_type"),
        }
        if status == 401:
            raise AuthenticationError(message, **kwargs)
        if status == 429:
            raise RateLimitError(message, **kwargs)
        raise APIStatusError(message, **kwargs)
    raise WarpLLMError(message, code=code)
