"""
Request models for the OpenAI-compatible chat endpoint.

Split out of api/app.py. Re-exported there, so `app.ChatRequest` and
`app.ChatMessage` keep working for every caller.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ChatMessage(BaseModel):
    """OpenAI-style message. `content` accepts a plain string OR a list of
    content blocks (multimodal format). Blocks are normalized to text —
    TokenMizer is a text proxy; non-text blocks (images) are dropped with
    their text parts preserved.

    Tool calling carries three more fields, all optional: an assistant
    turn's `tool_calls`, and a `role: "tool"` turn's `tool_call_id` (and
    OpenAI's optional `name`). They ride through the pipeline untouched
    and reach the provider adapter, which speaks them natively or
    translates them (see providers/tools.py)."""
    model_config = {"extra": "allow"}

    role: str
    content: str | list | None = ""
    tool_calls: list | None = None
    tool_call_id: str | None = None
    name: str | None = None

    def text(self) -> str:
        from tokenmizer.graph_memory.helpers import _content_to_text
        return _content_to_text(self.content)

    def to_dict(self) -> dict:
        """The pipeline's message shape: role + text content, plus the
        tool fields only when set, so an ordinary message is exactly the
        two-key dict every layer has always seen."""
        d: dict = {"role": self.role, "content": self.text()}
        if self.tool_calls:
            from tokenmizer.providers.tools import normalize_tool_calls
            d["tool_calls"] = normalize_tool_calls(self.tool_calls)
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        return d


class ChatRequest(BaseModel):
    """OpenAI-compatible request. Sampling params (temperature, top_p, stop)
    are forwarded to the provider. Unknown fields are accepted and ignored
    (extra='allow') so standard OpenAI clients never get a 422 — but only
    the fields below influence the call."""
    model_config = {"extra": "allow"}

    model: str | None = None
    # At least one message, as every provider requires. Without the bound an
    # empty list reached the provider twice (transformed, then untransformed
    # retry) and came back as a misleading 502 instead of a 422 naming the field.
    messages: list[ChatMessage] = Field(min_length=1)
    # Ranges every provider enforces anyway. Checked here so a bad value is
    # a 422 naming the field, not an opaque upstream 400 after the request
    # has been compressed, extracted and forwarded — `max_tokens: -5` went
    # all the way to the provider.
    max_tokens: int | None = Field(default=4096, ge=1)
    # OpenAI's current name for the same limit; newer SDK defaults and the
    # o-series reject `max_tokens` and send this instead. Without it the
    # client's limit was silently replaced by the 4096 default above.
    max_completion_tokens: int | None = Field(default=None, ge=1)
    stream: bool | None = False
    session_id: str | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    stop: str | list[str] | None = None
    # Tool/function calling, OpenAI shape. Forwarded to providers that
    # support it (see BaseProvider.supports_tools); a 501 otherwise.
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    parallel_tool_calls: bool | None = None
    # One choice is all this proxy ever returns. n > 1 was accepted and
    # silently answered with one, so a client indexing choices[1] crashed
    # far from the cause.
    n: int | None = Field(default=None, ge=1, le=1)

    @field_validator("session_id")
    @classmethod
    def _session_id_is_text(cls, v: str | None) -> str | None:
        # JSON accepts a lone surrogate ("\ud800"), and the id is used as a
        # key in SQLite and hashed for lock names. Unchecked it failed deep
        # in the ownership store and came back as a 503 "state unavailable"
        # — a client's bad input reported as the server's outage.
        if v is not None:
            try:
                v.encode("utf-8")
            except UnicodeEncodeError:
                raise ValueError("session_id must be valid Unicode text") from None
        return v


def _max_tokens(req: ChatRequest) -> int:
    """The completion limit the client asked for, under either name."""
    if req.max_completion_tokens is not None:
        return req.max_completion_tokens
    return req.max_tokens or 4096
