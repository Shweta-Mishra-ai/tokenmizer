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

# ── Error vocabulary ──────────────────────────────────────────────────────────
#
# Kept in step with the symptom vocabulary in hybrid_extractor: a failure the
# extractor is willing to emit should be a failure the validator recognises.
# Where the two disagreed, the extractor found the error and the validator
# silently dropped it ("corrupt header" scored 0.60 against a 0.65 threshold).

_EXCEPTION_NAME = re.compile(r'\b[A-Z][A-Za-z0-9]*(?:Error|Exception|Fault)\b')

# Build files with no extension — kept in step with hybrid_extractor's
# _FILE_EXTENSIONLESS, so a file the extractor is willing to emit is one the
# validator recognises as a filename.
_EXTENSIONLESS_FILE = re.compile(
    r'Dockerfile|Makefile|Procfile|Jenkinsfile|Gemfile|Rakefile|Vagrantfile|'
    r'Brewfile|Justfile|Caddyfile|CODEOWNERS|MANIFEST\.in'
)

# Vulnerability classes are named by acronym far more often than described.
_VULN_CLASS = re.compile(
    r'\b(?:IDOR|XSS|CSRF|SSRF|RCE|SQLi|TOCTOU|OOM|CVE-\d{4}-\d+)\b'
)

_ERROR_TERMS = (
    "error", "exception", "fail", "crash", "bug", "issue", "traceback",
    "422", "500", "404", "timeout", "timed out", "timing out", "null",
    "undefined", "corrupt", "malformed", "truncated", "deadlock", "race",
    "leak", "segfault", "segmentation fault", "panic", "hang", "flaky",
    "regression", "overflow", "locked", "denied", "refused", "unreachable",
    "mismatch", "invalid", "collision", "out of memory", "not triggering",
)

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
        source_role: Optional[str] = "assistant",
        extractor_confidence: float | None = None,
    ) -> ValidationResult:
        """
        Validate a candidate node.
        Returns ValidationResult with accepted=True/False and confidence score.

        source_role: "assistant" (default), "user", or None. Only "assistant"
        gets the small trust bonus below.

        The default stays "assistant" rather than None/no-bonus on
        purpose: every node type's confidence scoring and the 0.65
        min_confidence threshold were tuned assuming the bonus applies.
        Flipping the default to "no bonus" measurably regressed acceptance
        for dependency/task/decision nodes that have no source_role wired
        through yet — confirmed against the unit tests and the
        memory-accuracy fixture (tests/memory_accuracy/test_retention.py).
        Do not change it without re-validating that blast radius.

        Real attribution exists only for heuristic-extracted DECISIONS:
        HybridExtractor._extract_one_message knows which message (and role)
        each came from and threads it through, so a decision a USER stated
        does not get the assistant bonus. Every other node type, and
        LLM-synthesized decisions (no single-turn attribution), fall
        through to the default. Extending real role-tracking to them needs
        a broader ExtractedData schema change.

        extractor_confidence: the corroboration-based confidence computed by
        HybridExtractor.merge() (0.95 = both LLM and heuristic found it,
        0.80 = LLM-only, 0.65 = heuristic-only), when the candidate came
        through the extraction pipeline. Blended as:

            final = max(heuristic_score, (heuristic_score + extractor) / 2)

        Properties:
        - monotone: extractor evidence can only raise the score, never
          lower a label the heuristics already trust;
        - blend, not override: a corroborated candidate still needs
          non-junk heuristics to clear the threshold (heuristic-only 0.65
          does not automatically pass a 0.65 threshold);
        - hard rejects (noise labels/patterns) fire before this and are
          absolute.
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
        #
        # Every label over 20 chars gets a flat +0.10. Do NOT add a
        # length-based diminishing-returns tier here without a real
        # precision/recall harness: introducing one measurably reduced
        # task-extraction recall against the memory-accuracy fixture,
        # because several legitimately long, specific task labels lost
        # enough confidence to fall under the acceptance threshold.
        char_len = len(label)
        if char_len < 8:
            confidence -= 0.20
        elif char_len < 15:
            confidence -= 0.05
        elif char_len > 20:
            confidence += 0.10

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
        # ones — but only applied when actually known (see the
        # source_role note in validate()'s docstring).
        if source_role == "assistant":
            confidence += 0.05

        confidence = max(0.0, min(1.0, confidence))

        # Blend in the extractor's corroboration signal (see docstring)
        if extractor_confidence is not None:
            blended = (confidence + max(0.0, min(1.0, extractor_confidence))) / 2
            confidence = round(max(confidence, blended), 3)

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
        looks_like_a_path = bool(
            re.search(r'\.[a-z]{1,5}$', label, re.IGNORECASE)
            or "/" in label or "\\" in label
            or _EXTENSIONLESS_FILE.fullmatch(label)
        )
        if re.search(r'\.[a-z]{1,5}$', label, re.IGNORECASE):
            base += 0.25
        if "/" in label or "\\" in label:
            base += 0.10
        # Common filename words
        if any(w in label.lower() for w in ["main", "app", "config", "test", "model", "route", "api"]):
            base += 0.05
        # Short filenames are the UNAMBIGUOUS ones, and the generic
        # length/word-count penalties in validate() were rejecting exactly
        # those: `go.mod` scored 0.50 and `Dockerfile` 0.40 against a 0.65
        # threshold, while the longer `internal/store/postgres.go` sailed
        # through. Anything that is recognisably a filename clears the bar on
        # that evidence rather than on how many characters it happens to have.
        if looks_like_a_path:
            return max(base, 0.65)
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
        # A named exception class or vulnerability class IS the error, and is
        # the most precise form an error label can take. The generic
        # length/word-count penalties in validate() assume prose labels and
        # punish exactly that form: "ProxyError" scored 0.55 and was rejected,
        # while the vaguer "500 on an air-gapped host" scored 0.90. Give
        # identifier-shaped errors a floor that clears the threshold on their
        # own evidence rather than on sentence length.
        if _EXCEPTION_NAME.search(label) or _VULN_CLASS.search(label):
            return max(base + 0.15, 0.65)
        if any(t in label.lower() for t in _ERROR_TERMS):
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

    # A label is "essentially a path" when the path IS the label, not
    # merely its tail. Without this bound, `_FILE_PATTERN` (which anchors
    # on the end of the string) retyped any task or decision that happened
    # to finish with a filename — "User model in api/models.py" became a
    # FILE node, so it vanished from completed tasks and showed up as a
    # spurious file. Measured on the eval corpus, this alone cost several
    # points of task recall and of file precision.
    _MAX_PATH_LABEL_WORDS = 2

    def _check_type_mismatch(self, label: str, node_type: str) -> Optional[str]:
        """Return corrected type if we detect a mismatch, else None."""
        if (node_type != "file"
                and self._FILE_PATTERN.search(label)
                and "/" in label
                and len(label.split()) <= self._MAX_PATH_LABEL_WORDS):
            return "file"
        if node_type != "dependency" and self._DEP_PATTERN.match(label):
            return "dependency"
        if node_type != "endpoint" and self._URL_PATTERN.match(label):
            return "endpoint"
        return None


# ── Singleton ─────────────────────────────────────────────────────────────────

_validator: Optional[GraphValidator] = None


def get_validator(min_confidence: float | None = None) -> GraphValidator:
    """
    An explicit min_confidence returns a FRESH instance and must never
    touch the module-level singleton: overwriting it would change
    behaviour for every other caller that passes no argument, for the
    rest of the process lifetime.
    """
    if min_confidence is not None:
        return GraphValidator(min_confidence=min_confidence)

    global _validator
    if _validator is None:
        try:
            from tokenmizer.config.settings import get_settings
            threshold = get_settings().graph_checkpoint.min_confidence
        except Exception as e:
            # Low-risk fallback (unlike the auth.py case — this only
            # affects node-acceptance confidence threshold, not a security
            # control), but logged for debuggability: if a config error is
            # silently masking other settings too (see config/settings.py
            # fix), this log line is a clue something upstream is broken.
            logger.warning(
                f"Could not read min_confidence from settings ({e}) — "
                f"using hardcoded default 0.65"
            )
            threshold = 0.65
        _validator = GraphValidator(min_confidence=threshold)
    return _validator
