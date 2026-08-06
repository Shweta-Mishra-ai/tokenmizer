FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Layer caching for dependencies: install third-party deps from pyproject
# WITHOUT the package itself first (the tokenmizer/ source isn't copied yet —
# an editable/-e install here would fail because hatchling can't find the
# package dir). tomllib is stdlib in 3.11+.
COPY pyproject.toml README.md ./
RUN python -c "import tomllib; \
    deps = tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']; \
    print('\n'.join(deps))" > /tmp/reqs.txt \
    && pip install --no-cache-dir -r /tmp/reqs.txt

# Copy application code, then install the package itself (fast — deps cached)
COPY tokenmizer/ ./tokenmizer/
COPY tokenmizer.yaml ./
RUN pip install --no-cache-dir ".[anthropic,openai,gemini,cache]"

# Create directories that need to exist at runtime
RUN mkdir -p /app/checkpoints

# Pre-download the sentence-transformer model into a shared cache dir
# (done as root so the path is fixed, then handed to appuser).
#
# NON-FATAL on purpose, unlike the tiktoken bake below. The two look
# alike and are not:
#
#   tiktoken   — required, on the hot path of every request. If its
#                vocabulary is missing the proxy still answers, but every
#                token count silently drops to a char/4 estimate. Worth
#                failing the build over.
#   this model — the semantic cache only. Without it the cache degrades
#                to exact match, which is a slower cache, not a broken
#                image (semantic_cache/cache.py handles the absence).
#
# Hard-failing here made every image build depend on huggingface.co being
# reachable AND not rate-limiting, and it duly broke CI on a Hub hiccup.
# Retry once, then continue and say so in the build log.
ENV HF_HOME=/app/.hf-cache
RUN set -eu; \
    baked=0; \
    for attempt in 1 2; do \
      if python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"; then \
        baked=1; echo "embedding model baked in"; break; \
      fi; \
      echo "embedding model download failed (attempt $attempt of 2)"; \
      [ "$attempt" = 1 ] && sleep 15 || true; \
    done; \
    [ "$baked" = 1 ] || \
      echo "WARNING: no embedding model baked in — the semantic cache will run in exact-match mode"

# Pre-download tiktoken's BPE vocabulary at BUILD time.
#
# tiktoken ships no vocabulary — it fetches one from
# openaipublic.blob.core.windows.net the first time an encoding is
# used. Token counting is on the hot path of every proxied request, so
# leaving that as a runtime dependency means an air-gapped host, a
# restrictive egress policy, or a CDN outage degrades every request's
# token accounting (and, before the fix in core/tokenizer.py, failed
# the requests outright). Baking the vocabulary in makes the running
# container independent of that CDN.
ENV TIKTOKEN_CACHE_DIR=/app/.tiktoken-cache
RUN mkdir -p /app/.tiktoken-cache && python -c "\
import tiktoken; \
[tiktoken.get_encoding(e) for e in ('o200k_base', 'cl100k_base')]"

# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# IMPORTANT: --workers 1. Session locks, graph LRU cache, rate limiter and
# analytics are in-process. With >1 worker each process holds divergent state
# for the same session (last-writer-wins graph loss, split rate limits).
# Scale horizontally only after moving state to the Redis backend
# (TOKENMIZER_STATE_BACKEND=redis) AND making graph/analytics multi-process
# safe. Until then: one worker, scale vertically.
CMD ["python", "-m", "uvicorn", "tokenmizer.api.app:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
