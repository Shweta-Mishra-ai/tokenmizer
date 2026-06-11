#!/usr/bin/env bash
# TokenMizer — one-line setup
# curl -fsSL https://raw.githubusercontent.com/Shweta-Mishra-ai/tokenmizer/main/scripts/setup.sh | bash
set -e

BOLD="\033[1m"; GREEN="\033[32m"; YELLOW="\033[33m"; CYAN="\033[36m"; RESET="\033[0m"

echo ""
echo -e "${BOLD}🧠 TokenMizer Setup${RESET}"
echo -e "Never lose your AI context again."
echo "──────────────────────────────────"

# Python check
python3 -c "import sys; assert sys.version_info >= (3,10), 'Need Python 3.10+'" 2>/dev/null || {
  echo "❌ Python 3.10+ required. Install from https://python.org"; exit 1
}
echo -e "✅ Python $(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

# Install
echo -e "\n${CYAN}Installing TokenMizer...${RESET}"
pip install "tokenmizer[cache]" -q 2>/dev/null || pip3 install "tokenmizer[cache]" -q
echo -e "✅ TokenMizer installed"

# Detect provider
echo -e "\n${CYAN}Detecting provider...${RESET}"
PROVIDER=""; MODEL=""
if   [ -n "$ANTHROPIC_API_KEY" ] || [ -n "$TOKENMIZER_ANTHROPIC_API_KEY" ]; then
  PROVIDER="anthropic"; MODEL="claude-sonnet-4-6"; echo -e "✅ Anthropic detected"
elif [ -n "$OPENAI_API_KEY" ] || [ -n "$TOKENMIZER_OPENAI_API_KEY" ]; then
  PROVIDER="openai"; MODEL="gpt-4o"; echo -e "✅ OpenAI detected"
elif [ -n "$DEEPSEEK_API_KEY" ] || [ -n "$TOKENMIZER_DEEPSEEK_API_KEY" ]; then
  PROVIDER="deepseek"; MODEL="deepseek-chat"; echo -e "✅ DeepSeek detected"
elif curl -sf http://localhost:11434/api/tags &>/dev/null; then
  PROVIDER="ollama"; MODEL="llama3"; echo -e "✅ Ollama detected (free, local)"
else
  echo -e "${YELLOW}⚠️  No provider found. Install Ollama (free): https://ollama.ai${RESET}"
  PROVIDER="anthropic"; MODEL="claude-sonnet-4-6"
fi

# Write config
if [ ! -f "tokenmizer.yaml" ]; then
  cat > tokenmizer.yaml << YAML
provider: ${PROVIDER}
default_model: ${MODEL}
graph_checkpoint:
  enabled: true
  trigger_at_percent: 0.85
  storage_dir: ./checkpoints
compression:
  enabled: true
cache:
  enabled: true
state_backend: memory
YAML
  echo -e "✅ Config written: tokenmizer.yaml"
else
  echo -e "✅ Config exists — skipping"
fi

# Claude Code MCP (if claude is installed)
if command -v claude &>/dev/null && [ ! -f ".mcp.json" ]; then
  cat > .mcp.json << JSON
{
  "mcpServers": {
    "tokenmizer": {
      "command": "python",
      "args": ["-m", "tokenmizer.mcp.server"],
      "env": { "TOKENMIZER_URL": "http://localhost:8000" }
    }
  }
}
JSON
  echo -e "✅ .mcp.json written — Claude Code MCP ready"
fi

echo ""
echo -e "${GREEN}${BOLD}✅ Ready!${RESET}"
echo ""
echo -e "  Start:     ${CYAN}tokenmizer serve${RESET}"
echo -e "  Dashboard: ${CYAN}http://localhost:8000${RESET}"
echo -e "  Docs:      ${CYAN}http://localhost:8000/docs${RESET}"
[ "$PROVIDER" = "ollama" ] && echo -e "\n  ${YELLOW}Tip: run 'ollama pull llama3' first if not done${RESET}"
echo ""
