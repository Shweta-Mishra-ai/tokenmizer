"""Shared error types for TokenMizer."""
from __future__ import annotations


class TokenMizerError(Exception):
    """Base exception."""


class ProviderError(TokenMizerError):
    def __init__(self, provider: str, error_type: str, message: str,
                 retryable: bool = False, retry_after: float = 0.0):
        self.provider = provider
        self.error_type = error_type
        self.retryable = retryable
        self.retry_after = retry_after
        super().__init__(f"[{provider}] {error_type}: {message}")


class ConfigError(TokenMizerError):
    """Invalid configuration."""


class StorageError(TokenMizerError):
    """Persistence failure."""
