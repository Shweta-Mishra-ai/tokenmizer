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


class StorageError(TokenMizerError):
    """Persistence failure."""


class CheckpointPersistError(StorageError):
    """Checkpoint write failed. Callers MUST NOT treat a checkpoint as
    successfully created if this is raised — there is no fallback write
    path, so a swallowed instance of this error means data loss."""


# There is deliberately no GraphPersistError. A graph write that fails
# does NOT raise: persistence.py sets _persistence_broken, surfaces it
# through /health and /api/stats, and the turn continues — the caller came
# for an answer, and a lost node must be visible rather than fatal. An
# exception class nothing raises reads as a contract that does not exist,
# which is how the deleted one was read.
