"""
Graph Validator & Confidence Scorer
tokenmizer/graph_memory/validator.py

Every node candidate goes through this pipeline before touching the graph:

    raw extracted data
         ↓
    confidence score (0.0–1.0)
         ↓
    validation (reject noise, too-short labels, generic words)
         ↓
    type correction (wrong type assigned by extractor)
         ↓
    accepted / rejected

This is the quality gate your friend identified as missing.
Without this, graph pollution accumulates every session.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ── Noise patterns — these should never become nodes ─────────────────────────

_NOISE_LABELS = frozenset({
    "this", "that", "it", "the", "a", "an", "yes", "no", "ok", "okay",
    "sure", "done", "good", "great", "thanks", "thank you", "please",
    "here", "there", "something", "anything", "nothing", "everything",
    "mentioned", "above", "below", "following", "previous",
    "true", "false", "none", "null", "undefined",
    "error", "issue", "problem", "thing", "stuff", "item",
})

_NOISE_PATTERNS = [
    re.compile(r"^\d+$"),                          # pure numbers
    re.compile(r"^[^a-zA-Z]+$"),                   # no letters at all
    re.compile(r"^.{1,3}$"),                        # too short (≤3 chars)
    re.compile(r"^\s+$"),                           # whitespace only
    re.compile(r"^https?://", re.IGNORECASE),       # URLs (not useful as labels)
]

# Generic single-word labels that carry no information about THIS project
_GENERIC_SINGLE_WORDS = frozenset({
    "implement", "create", "update", "fix", "add", "remove", "delete",
    "build", "test", "check", "review", "refactor", "debug", "deploy",
    "write", "read", "get", "set", "run", "start", "stop", "init",
    "setup", "configure", "install", "upgrade", "migrate", "generate",
})


# ── Confidence signals ────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    accepted: bool
    confidence: float           # 0.0–1.0
    rejection_reason: str = ""  # non-empty if rejected
    corrected_type: Optional[str] = None  # if type was wrong


class GraphValidator:
    """
    Validates and scores node candidates before graph insertion.
    
    Confidence scoring:
      - starts at 0.5 (neutral)
      - boosted by: specific language, file paths, keywords, context
      - penalised by: short labels, generic words, noise patterns
    
    Threshold: accept if confidence >= 0.5 (configurable)
    """

    def __init__(self, min_confidence: float = 0.65):
        self.min_confidence = min_confidence

    def validate(
        self,
        label: str,
        node_type: str,
        summary: str = "",
        source_role: str = "assistant",
    ) -> ValidationResult:
        """
        Validate a candidate node.
        Returns ValidationResult with accepted=True/False and confidence score.
        """
        label = label.strip()

        # ── Hard rejects ─────────────────────────────────────────────────────

        if not label:
            return ValidationResult(False, 0.0, "empty label")

        label_lower = label.lower().strip(".,!?")

        if label_lower in _NOISE_LABELS:
            return ValidationResult(False, 0.0, f"noise word: {label_lower!r}")

        for pat in _NOISE_PATTERNS:
            if pat.match(label):
                return ValidationResult(False, 0.0, f"noise pattern: {label!r}")

        # Single generic verb with no object
        words = label_lower.split()
        if len(words) == 1 and label_lower in _GENERIC_SINGLE_WORDS:
            return ValidationResult(False, 0.0, f"generic single verb: {label_lower!r}")

        # ── Confidence scoring ────────────────────────────────────────────────

        confidence = 0.50  # baseline

        # Length signals: longer = more specific = higher confidence
        char_len = len(label)
        if char_len < 8:
            confidence -= 0.20
        elif char_len < 15:
            confidence -= 0.05
        elif char_len > 20:
            confidence += 0.10
        elif char_len > 40:
            confidence += 0.05  # diminishing returns on very long labels

        # Word count: 2–8 words is the sweet spot
        word_count = len(words)
        if word_count == 1:
            confidence -= 0.10
        elif 2 <= word_count <= 8:
            confidence += 0.10
        elif word_count > 15:
            confidence -= 0.10  # too long, probably noise

        # Type-specific signals
        if node_type == "file":
            confidence = self._score_file(label, confidence)
        elif node_type == "decision":
            confidence = self._score_decision(label, summary, confidence)
        elif node_type == "task":
            confidence = self._score_task(label, confidence)
        elif node_type == "error":
            confidence = self._score_error(label, confidence)
        elif node_type == "environment":
            confidence = self._score_environment(label, confidence)
        elif node_type == "goal":
            confidence = self._score_goal(label, confidence)
        elif node_type == "dependency":
            confidence = self._score_dependency(label, confidence)

        # Summary bonus: if extractor provided rationale, trust more
        if summary and len(summary) > 10:
            confidence += 0.08

        # Source role: assistant claims are generally more reliable than user
        if source_role == "assistant":
            confidence += 0.05

        confidence = max(0.0, min(1.0, confidence))

        # ── Final decision ────────────────────────────────────────────────────

        if confidence < self.min_confidence:
            return ValidationResult(
                False, confidence,
                f"confidence {confidence:.2f} < threshold {self.min_confidence}"
            )

        # Check for possible type mismatch
        corrected = self._check_type_mismatch(label, node_type)

        return ValidationResult(
            accepted=True,
            confidence=round(confidence, 3),
            corrected_type=corrected,
        )

    # ── Type-specific scorers ─────────────────────────────────────────────────

    def _score_file(self, label: str, base: float) -> float:
        # File paths are high confidence if they have an extension or /
        if re.search(r'\.[a-z]{1,5}$', label, re.IGNORECASE):
            base += 0.25
        if "/" in label or "\\" in label:
            base += 0.10
        # Common filename words
        if any(w in label.lower() for w in ["main", "app", "config", "test", "model", "route", "api"]):
            base += 0.05
        return base

    def _score_decision(self, label: str, summary: str, base: float) -> float:
        # Decisions need rationale — without summary they're weaker
        if not summary:
            base -= 0.10
        # Decision language is a strong signal
        decision_words = ["use", "using", "chose", "decided", "switch", "instead", "because", "over"]
        if any(w in label.lower() for w in decision_words):
            base += 0.15
        # Technology names are strong decision signals
        tech_names = ["postgresql", "redis", "sqlite", "mongodb", "jwt", "oauth",
                      "docker", "kubernetes", "fastapi", "django", "flask",
                      "react", "vue", "angular", "typescript", "python"]
        if any(t in label.lower() for t in tech_names):
            base += 0.12
        return base

    def _score_task(self, label: str, base: float) -> float:
        # Tasks should be verb + object
        action_verbs = ["implement", "create", "add", "fix", "update", "write",
                        "build", "configure", "set up", "refactor", "test"]
        has_verb = any(label.lower().startswith(v) or f" {v} " in label.lower()
                      for v in action_verbs)
        if has_verb:
            base += 0.12
        # Very generic tasks are low value
        if label.lower() in {"fix bug", "add tests", "update code", "make changes"}:
            base -= 0.20
        return base

    def _score_error(self, label: str, base: float) -> float:
        error_terms = ["error", "exception", "fail", "crash", "bug", "issue",
                       "traceback", "422", "500", "404", "timeout", "null", "undefined"]
        if any(t in label.lower() for t in error_terms):
            base += 0.15
        return base

    def _score_environment(self, label: str, base: float) -> float:
        # Version numbers are strong environment signals
        if re.search(r'\d+\.\d+', label):
            base += 0.20
        env_terms = ["python", "node", "npm", "pip", "docker", "postgres",
                     "redis", "ubuntu", "linux", "macos", "windows", "aws", "gcp", "azure"]
        if any(t in label.lower() for t in env_terms):
            base += 0.15
        return base

    def _score_goal(self, label: str, base: float) -> float:
        # Goals need to describe a system/product/outcome
        if len(label) < 15:
            base -= 0.20  # "fix bug" is not a goal
        build_verbs = ["build", "create", "develop", "implement", "design"]
        if any(v in label.lower() for v in build_verbs):
            base += 0.15
        return base

    def _score_dependency(self, label: str, base: float) -> float:
        # Package names with versions are strong
        if re.search(r'[>=<]+\s*[\d.]+', label):
            base += 0.20
        # Known package ecosystems
        if re.match(r'^[a-z][a-z0-9\-_]+$', label, re.IGNORECASE):
            base += 0.10  # looks like a package name
        return base

    # ── Type mismatch detection ───────────────────────────────────────────────

    _FILE_PATTERN = re.compile(r'\.[a-zA-Z]{1,5}$')
    _DEP_PATTERN = re.compile(r'^[a-z][a-z0-9\-_]+(==|>=|<=|~=|>|<)\d', re.IGNORECASE)
    _URL_PATTERN = re.compile(r'^(GET|POST|PUT|DELETE|PATCH)\s+/')

    def _check_type_mismatch(self, label: str, node_type: str) -> Optional[str]:
        """Return corrected type if we detect a mismatch, else None."""
        if node_type != "file" and self._FILE_PATTERN.search(label) and "/" in label:
            return "file"
        if node_type != "dependency" and self._DEP_PATTERN.match(label):
            return "dependency"
        if node_type != "endpoint" and self._URL_PATTERN.match(label):
            return "endpoint"
        return None


# ── Singleton ─────────────────────────────────────────────────────────────────

_validator: Optional[GraphValidator] = None


def get_validator(min_confidence: float = 0.65) -> GraphValidator:
    global _validator
    if _validator is None:
        _validator = GraphValidator(min_confidence=min_confidence)
    return _validator
