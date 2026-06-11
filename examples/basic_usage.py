"""
TokenMizer usage examples.

1. Basic drop-in proxy (same as OpenAI client)
2. Session resume — pick up where you left off
3. Manual checkpoint + resume
"""
import asyncio

# ─────────────────────────────────────────────────────────────────────────────
# Example 1: Drop-in OpenAI-compatible proxy
# Just change base_url — everything else stays the same
# ─────────────────────────────────────────────────────────────────────────────

def example_openai_client():
    """Works with any OpenAI-compatible SDK client."""
    from openai import OpenAI

    client = OpenAI(
        api_key="your-api-key",          # your provider API key
        base_url="http://localhost:8000/v1",  # point at TokenMizer
    )

    response = client.chat.completions.create(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "Write a Python function to reverse a string"}],
        extra_body={"session_id": "my-coding-session"},  # enables graph memory
    )
    print(response.choices[0].message.content)
    print(f"Tokens saved: {response.model_extra.get('tokenmizer', {}).get('total_saved', 0)}")


# ─────────────────────────────────────────────────────────────────────────────
# Example 2: Session resume with httpx
# ─────────────────────────────────────────────────────────────────────────────

async def example_session_resume():
    import httpx

    BASE = "http://localhost:8000"
    SESSION_ID = "my-project-session"
    HEADERS = {}  # add {"Authorization": "Bearer <key>"} if auth is enabled

    async with httpx.AsyncClient() as client:

        # --- Turn 1 ---
        r = await client.post(f"{BASE}/v1/chat/completions", json={
            "model": "claude-sonnet-4-6",
            "session_id": SESSION_ID,
            "messages": [
                {"role": "user", "content": "Let's build a task management API with FastAPI and PostgreSQL"},
            ],
        }, headers=HEADERS)
        turn1 = r.json()
        print("Turn 1:", turn1["choices"][0]["message"]["content"][:200])

        # --- Turn 2 ---
        r = await client.post(f"{BASE}/v1/chat/completions", json={
            "model": "claude-sonnet-4-6",
            "session_id": SESSION_ID,
            "messages": [
                {"role": "user", "content": "Let's build a task management API with FastAPI and PostgreSQL"},
                {"role": "assistant", "content": turn1["choices"][0]["message"]["content"]},
                {"role": "user", "content": "Start with the Task model — id, title, description, status, created_at"},
            ],
        }, headers=HEADERS)
        turn2 = r.json()
        print("Turn 2:", turn2["choices"][0]["message"]["content"][:200])

        # --- Manual checkpoint ---
        ckpt_r = await client.post(
            f"{BASE}/api/checkpoint?session_id={SESSION_ID}",
            headers=HEADERS,
        )
        ckpt = ckpt_r.json()
        print(f"\nCheckpoint created: {ckpt['checkpoint_id']}")
        print(f"Resume tokens: {ckpt['resume_tokens']}")
        print(f"\nResume block:\n{ckpt['resume_standard']}")

        # --- New session — resume from checkpoint ---
        resume_r = await client.get(
            f"{BASE}/api/resume/{SESSION_ID}?level=standard",
            headers=HEADERS,
        )
        resume_data = resume_r.json()
        resume_ctx = resume_data["resume_context"]

        print(f"\n--- Starting new session with {resume_data['token_count']}-token resume ---")

        new_session_r = await client.post(f"{BASE}/v1/chat/completions", json={
            "model": "claude-sonnet-4-6",
            "session_id": "my-project-session-resumed",
            "messages": [
                {
                    "role": "system",
                    "content": f"[Previous session context]\n{resume_ctx}\n\nContinue from where we left off.",
                },
                {
                    "role": "user",
                    "content": "Now add the endpoints: GET /tasks, POST /tasks, PUT /tasks/{id}, DELETE /tasks/{id}",
                },
            ],
        }, headers=HEADERS)

        new_session = new_session_r.json()
        print("Resumed:", new_session["choices"][0]["message"]["content"][:300])


# ─────────────────────────────────────────────────────────────────────────────
# Example 3: Direct Python API (without HTTP server)
# ─────────────────────────────────────────────────────────────────────────────

async def example_direct_api():
    """Use TokenMizer graph memory directly in your Python app."""
    import tempfile

    from tokenmizer.checkpoints.manager import CheckpointManager
    from tokenmizer.graph_memory.graph import GraphMemory

    with tempfile.TemporaryDirectory() as tmp:
        graph = GraphMemory("my-session", storage_dir=tmp)
        mgr = CheckpointManager(storage_dir=tmp)

        # Simulate a conversation being processed
        messages = [
            {"role": "user", "content": "Build a REST API for a blog with FastAPI"},
            {"role": "assistant", "content": "Completed: project setup. Created main.py, models.py, routes/posts.py. Decided: PostgreSQL for storage."},
            {"role": "user", "content": "Add authentication"},
            {"role": "assistant", "content": "Implemented JWT auth. Working on: refresh token rotation."},
        ]

        # Update graph from messages
        graph.extract_from_messages(messages, incremental=True)

        # See what was extracted
        stats = graph.stats()
        print(f"Graph: {stats['node_count']} nodes, {stats['by_type']}")

        # Get context block for injection
        ctx = graph.to_context_block(token_budget=300)
        print(f"\nContext block ({len(ctx.split())} words):\n{ctx}")

        # Create checkpoint
        ckpt = mgr.create(
            session_id="my-session",
            messages=messages,
            graph=graph,
            context_pct=0.87,
        )
        print(f"\nCheckpoint: {ckpt.resume_tokens} resume tokens")
        print(f"Resume (standard):\n{ckpt.resume_standard}")


if __name__ == "__main__":
    print("Example 3: Direct API\n")
    asyncio.run(example_direct_api())
