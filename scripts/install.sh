#!/usr/bin/env bash
# TokenMizer — Universal One-Line Installer
# Usage: curl -fsSL https://raw.githubusercontent.com/Shweta-Mishra-ai/tokenmizer/main/scripts/install.sh | bash
#
# Supports: macOS, Linux (Ubuntu/Debian/Fedora/Arch), Windows (WSL)
# Installs: Python package + auto-detects provider + writes config + MCP setup

set -euo pipefail

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
ok()   { echo -e "${GREEN}✅ $*${RESET}"; }
warn() { echo -e "${YELLOW}⚠️  $*${RESET}"; }
err()  { echo -e "${RED}❌ $*${RESET}"; exit 1; }
info() { echo -e "${CYAN}   $*${RESET}"; }

echo ""
echo -e "${BOLD}🧠 TokenMizer Installer${RESET}"
echo -e "   Never lose your AI context again."
echo "   ─────────────────────────────────"

# ── OS detection ──────────────────────────────────────────────────────────────
OS="unknown"
case "$(uname -s)" in
  Linux*)  OS="linux" ;;
  Darwin*) OS="macos" ;;
  CYGWIN*|MINGW*|MSYS*) OS="windows" ;;
esac
info "Detected OS: $OS"

# ── Python check + auto-install ───────────────────────────────────────────────
PYTHON=""
for cmd in python3.12 python3.11 python3.10 python3 python; do
  if command -v "$cmd" &>/dev/null; then
    ver=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
    major="${ver%%.*}"; minor="${ver#*.}"
    if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ] 2>/dev/null; then
      PYTHON="$cmd"; break
    fi
  fi
done

if [ -z "$PYTHON" ]; then
  warn "Python 3.10+ not found. Attempting install..."
  if   [ "$OS" = "linux" ] && command -v apt-get &>/dev/null; then
    sudo apt-get update -qq && sudo apt-get install -y python3 python3-pip python3-venv
    PYTHON="python3"
  elif [ "$OS" = "linux" ] && command -v dnf &>/dev/null; then
    sudo dnf install -y python3 python3-pip
    PYTHON="python3"
  elif [ "$OS" = "linux" ] && command -v pacman &>/dev/null; then
    sudo pacman -S --noconfirm python python-pip
    PYTHON="python3"
  elif [ "$OS" = "macos" ]; then
    if command -v brew &>/dev/null; then
      brew install python@3.12
      PYTHON="python3.12"
    else
      err "Install Homebrew first: https://brew.sh, then re-run this script."
    fi
  else
    err "Cannot auto-install Python. Please install Python 3.10+ from https://python.org"
  fi
fi
ok "Python $($PYTHON --version 2>&1 | awk '{print $2}')"

# ── pip check ─────────────────────────────────────────────────────────────────
if ! $PYTHON -m pip --version &>/dev/null 2>&1; then
  warn "pip not found — installing..."
  if [ "$OS" = "linux" ] && command -v apt-get &>/dev/null; then
    sudo apt-get install -y python3-pip -qq
  else
    $PYTHON -m ensurepip --upgrade || err "pip install failed. Run: $PYTHON -m ensurepip"
  fi
fi
ok "pip ready"

# ── Install TokenMizer ────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}Installing TokenMizer...${RESET}"

# Detect what extras to install based on available providers
EXTRAS="cache"  # always include semantic cache

if [ -n "${ANTHROPIC_API_KEY:-}" ] || [ -n "${TOKENMIZER_ANTHROPIC_API_KEY:-}" ]; then
  EXTRAS="$EXTRAS,anthropic"
fi
if [ -n "${OPENAI_API_KEY:-}" ] || [ -n "${TOKENMIZER_OPENAI_API_KEY:-}" ]; then
  EXTRAS="$EXTRAS,openai"
fi
if [ -n "${GEMINI_API_KEY:-}" ] || [ -n "${TOKENMIZER_GEMINI_API_KEY:-}" ]; then
  EXTRAS="$EXTRAS,gemini"
fi
if [ -n "${COHERE_API_KEY:-}" ] || [ -n "${TOKENMIZER_COHERE_API_KEY:-}" ]; then
  EXTRAS="$EXTRAS,cohere"
fi

info "Installing with extras: [$EXTRAS]"
$PYTHON -m pip install "tokenmizer[$EXTRAS]" --quiet --upgrade \
  || err "Install failed. Try: $PYTHON -m pip install tokenmizer"
ok "TokenMizer installed"

# ── Provider detection ────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}Detecting AI provider...${RESET}"
PROVIDER=""; MODEL=""; PROVIDER_KEY_VAR=""

detect_provider() {
  if   [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    PROVIDER="anthropic"; MODEL="claude-sonnet-4-6"; PROVIDER_KEY_VAR="ANTHROPIC_API_KEY"
  elif [ -n "${TOKENMIZER_ANTHROPIC_API_KEY:-}" ]; then
    PROVIDER="anthropic"; MODEL="claude-sonnet-4-6"; PROVIDER_KEY_VAR="TOKENMIZER_ANTHROPIC_API_KEY"
  elif [ -n "${OPENAI_API_KEY:-}" ]; then
    PROVIDER="openai"; MODEL="gpt-4o"; PROVIDER_KEY_VAR="OPENAI_API_KEY"
  elif [ -n "${TOKENMIZER_OPENAI_API_KEY:-}" ]; then
    PROVIDER="openai"; MODEL="gpt-4o"; PROVIDER_KEY_VAR="TOKENMIZER_OPENAI_API_KEY"
  elif [ -n "${GEMINI_API_KEY:-}" ] || [ -n "${TOKENMIZER_GEMINI_API_KEY:-}" ]; then
    PROVIDER="gemini"; MODEL="gemini-1.5-pro"; PROVIDER_KEY_VAR="GEMINI_API_KEY"
  elif [ -n "${DEEPSEEK_API_KEY:-}" ] || [ -n "${TOKENMIZER_DEEPSEEK_API_KEY:-}" ]; then
    PROVIDER="deepseek"; MODEL="deepseek-chat"; PROVIDER_KEY_VAR="DEEPSEEK_API_KEY"
  elif [ -n "${MISTRAL_API_KEY:-}" ] || [ -n "${TOKENMIZER_MISTRAL_API_KEY:-}" ]; then
    PROVIDER="mistral"; MODEL="mistral-large-latest"; PROVIDER_KEY_VAR="MISTRAL_API_KEY"
  elif [ -n "${GROK_API_KEY:-}" ] || [ -n "${TOKENMIZER_GROK_API_KEY:-}" ]; then
    PROVIDER="grok"; MODEL="grok-2"; PROVIDER_KEY_VAR="GROK_API_KEY"
  elif [ -n "${COHERE_API_KEY:-}" ] || [ -n "${TOKENMIZER_COHERE_API_KEY:-}" ]; then
    PROVIDER="cohere"; MODEL="command-r-plus"; PROVIDER_KEY_VAR="COHERE_API_KEY"
  elif [ -n "${OPENROUTER_API_KEY:-}" ] || [ -n "${TOKENMIZER_OPENROUTER_API_KEY:-}" ]; then
    PROVIDER="openrouter"; MODEL="openai/gpt-4o"; PROVIDER_KEY_VAR="OPENROUTER_API_KEY"
  elif curl -sf http://localhost:11434/api/tags &>/dev/null 2>&1; then
    PROVIDER="ollama"; MODEL="llama3"; PROVIDER_KEY_VAR=""
  fi
}

detect_provider

if [ -z "$PROVIDER" ]; then
  warn "No API key found in environment."
  echo ""
  echo "  Options:"
  info "  a) Set key now:   export ANTHROPIC_API_KEY=sk-ant-..."
  info "  b) Use Ollama (free, local): https://ollama.ai"
  info "  c) Press Enter to configure manually after install"
  echo ""
  read -r -p "   Enter your API key (or press Enter to skip): " USER_KEY
  if [ -n "$USER_KEY" ]; then
    # Detect provider from key prefix
    case "$USER_KEY" in
      sk-ant-*)   PROVIDER="anthropic"; MODEL="claude-sonnet-4-6" ;;
      sk-proj-*|sk-[a-zA-Z0-9]*) PROVIDER="openai"; MODEL="gpt-4o" ;;
      AIza*)      PROVIDER="gemini"; MODEL="gemini-1.5-pro" ;;
      *)          PROVIDER="openai"; MODEL="gpt-4o" ;;  # default
    esac
    export TOKENMIZER_${PROVIDER^^}_API_KEY="$USER_KEY"
    info "Provider set to: $PROVIDER"
  else
    warn "No provider configured. Edit tokenmizer.yaml manually after install."
    PROVIDER="anthropic"; MODEL="claude-sonnet-4-6"
  fi
else
  ok "$PROVIDER detected (model: $MODEL)"
fi

# ── Write config ──────────────────────────────────────────────────────────────
CONFIG_FILE="tokenmizer.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
  cat > "$CONFIG_FILE" << YAML
# TokenMizer Configuration
# Docs: https://github.com/Shweta-Mishra-ai/tokenmizer

provider: ${PROVIDER}
default_model: ${MODEL}

graph_checkpoint:
  enabled: true
  trigger_at_percent: 0.85        # checkpoint at 85% context usage
  storage_dir: ~/.tokenmizer/checkpoints
  use_llm_extraction: false       # set true for 90%+ recall (uses haiku/gpt-4o-mini)

compression:
  enabled: true
  engine: heuristic               # heuristic | llmlingua2

cache:
  enabled: true
  max_size: 10000

memory:
  max_tokens_before_summary: 8000
  recent_turns_verbatim: 6

rate_limit:
  requests_per_minute: 60
  burst: 10

state_backend: memory             # memory (dev) | redis (prod)
YAML
  ok "Config written: $CONFIG_FILE"
else
  info "Config exists — skipping (delete $CONFIG_FILE to reset)"
fi

# ── Claude Code MCP setup ─────────────────────────────────────────────────────
if command -v claude &>/dev/null; then
  MCP_FILE="${CLAUDE_CONFIG_DIR:-$HOME/.claude}/settings.json"
  if [ ! -f "$MCP_FILE" ] || ! grep -q "tokenmizer" "$MCP_FILE" 2>/dev/null; then
    mkdir -p "$(dirname "$MCP_FILE")"
    # Merge into existing settings or create new
    if [ -f "$MCP_FILE" ]; then
      # Backup
      cp "$MCP_FILE" "${MCP_FILE}.bak"
      warn "Backed up existing settings to ${MCP_FILE}.bak"
    fi
    # FIXED: this hardcoded "python" instead of using the $PYTHON variable
    # already detected earlier in this same script (line ~33) — meaning the
    # careful python3/python3.11/python3.10 detection done above was wasted;
    # the .mcp.json it wrote would break on any Debian/Ubuntu box where only
    # python3 exists, exactly the case this script otherwise handles correctly.
    cat > .mcp.json << JSON
{
  "mcpServers": {
    "tokenmizer": {
      "command": "${PYTHON}",
      "args": ["-m", "tokenmizer.mcp.server"],
      "env": { "TOKENMIZER_URL": "http://localhost:8000" }
    }
  }
}
JSON
    ok "Claude Code MCP configured (.mcp.json)"
  else
    info "Claude Code MCP already configured"
  fi
fi

# ── Cursor / Continue.dev hint ────────────────────────────────────────────────
if [ -d "$HOME/.cursor" ] || [ -d "$HOME/Library/Application Support/Cursor" ]; then
  info "Cursor detected! Set API Base URL to: http://localhost:8000/v1"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}✅ TokenMizer installed successfully!${RESET}"
echo ""
echo -e "  ${BOLD}Start server:${RESET}"
echo -e "    ${CYAN}tokenmizer serve${RESET}"
echo ""
echo -e "  ${BOLD}Or with Docker:${RESET}"
echo -e "    ${CYAN}docker-compose up${RESET}"
echo ""
echo -e "  ${BOLD}Endpoints (after start):${RESET}"
echo -e "    Proxy:     ${CYAN}http://localhost:8000/v1/chat/completions${RESET}"
echo -e "    Dashboard: ${CYAN}http://localhost:8000${RESET}"
echo -e "    Docs:      ${CYAN}http://localhost:8000/docs${RESET}"
echo ""
if [ "$PROVIDER" = "ollama" ]; then
  echo -e "  ${YELLOW}Tip: run 'ollama pull llama3' first if not done${RESET}"
fi
echo -e "  ${BOLD}Docs:${RESET} https://github.com/Shweta-Mishra-ai/tokenmizer"
echo ""
