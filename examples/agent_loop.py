"""
Plain agent-loop example — no framework, just tokenmizer.agents.Memory.

Shows the shape every other adapter in this directory builds on: add()
what happened, context() for what to inject into the next prompt,
search() when you need one specific fact. No server, no API key, no
network required to run this file — call_model is swappable for any
provider's SDK.
"""
from __future__ import annotations

from typing import Callable

from tokenmizer.agents import Memory

CallModel = Callable[[str, str], str]


def run_turn(memory: Memory, user_message: str, call_model: CallModel) -> str:
    """One turn of an agent loop.

    `call_model(system_prompt, user_message) -> reply` is the caller's own
    LLM call. The turn is recorded into `memory` after the reply comes
    back, so a call that raises leaves nothing half-written.
    """
    system_prompt = "You are a helpful engineering assistant."
    resume = memory.context(token_budget=400)
    if resume:
        system_prompt = f"{system_prompt}\n\n[Session context]\n{resume}"

    reply = call_model(system_prompt, user_message)

    memory.add([
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": reply},
    ])
    return reply


def main() -> None:
    import tempfile

    def fake_model(system_prompt: str, user_message: str) -> str:
        # Swap this for a real provider call — Anthropic, OpenAI, whatever
        # the rest of your agent already uses. TokenMizer's Memory does
        # not need to know which one.
        if "database" in user_message.lower():
            return "Decided: use PostgreSQL for order storage."
        return "Understood."

    with tempfile.TemporaryDirectory() as tmp:
        memory = Memory("demo-agent-loop", storage_dir=tmp)
        print(run_turn(
            memory,
            "We need to build the checkout service. What database should we use?",
            fake_model,
        ))
        print(run_turn(memory, "Go with your last recommendation.", fake_model))
        print("\nSession context so far:\n" + memory.context())


if __name__ == "__main__":
    main()
