# -*- coding: utf-8 -*-
"""Generate docs/assets/graph-demo.gif — the graph explorer, in motion.

The three still screenshots in the README show what the graph looks like.
They cannot show the thing that makes it worth opening: that it is one
self-contained page you *drive* — switch layout, filter a type, click a
node for its provenance — with no server behind it.

Nothing here is staged. The page is built by the shipped
`to_share_html()` from a real labelled-corpus session, opened from a
`file://` URL exactly as a reader would open a shared file, and driven
through the same controls a reader clicks. If the product regresses, so
does this GIF.

    python scripts/gen_graph_demo.py [output_path] [session_id]

Needs Playwright with Chromium (`pip install playwright && playwright
install chromium`) and Pillow. Both are dev-only; nothing in the package
imports this.
"""
from __future__ import annotations

import asyncio
import json
import os
import pathlib
import sys
import tempfile

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                   else "docs/assets/graph-demo.gif")
SESSION = sys.argv[2] if len(sys.argv) > 2 else "fastapi_auth"

# The same viewport as the still screenshots in docs/assets. Shorter and
# narrower were both tried: the timeline needs the height (every node on
# one diagonal, so the lanes pile up without it) and the toolbar wraps to
# two rows below 1500. Downscaled to _TARGET_W afterwards, so this costs
# layout quality, not file size.
VIEWPORT = {"width": 1500, "height": 940}

# (label, milliseconds to hold this frame). The hold times are the point:
# a reader needs long enough on each view to read it, and the transitions
# between them are what shows the page is interactive rather than a
# picture. GIF delays are centiseconds, so these are multiples of 10.
_HOLD_LONG = 2200
_HOLD_SHORT = 1400

# The width the README's <img> tag renders this at. Matching it exactly
# avoids both upscaling blur and paying for pixels nobody sees.
_TARGET_W = 900


def build_page() -> pathlib.Path:
    """Render the shipped share HTML for a corpus session to a temp file."""
    from tokenmizer.graph_memory.graph import GraphMemory
    from tokenmizer.graph_memory.visualization import to_share_html

    corpus = pathlib.Path("benchmarks/eval/corpus") / f"{SESSION}.json"
    if not corpus.exists():
        raise SystemExit(
            f"No corpus session at {corpus}. Pass one of: "
            + ", ".join(sorted(p.stem for p in corpus.parent.glob("*.json")))
        )
    messages = json.loads(corpus.read_text())["messages"]

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="tokenmizer-demo-"))
    graph = GraphMemory(SESSION, storage_dir=str(tmp))
    graph.extract_from_messages(messages, incremental=False)
    if not graph._nodes:
        raise SystemExit(f"{SESSION} extracted no nodes — nothing to show.")

    page = tmp / "graph.html"
    page.write_text(to_share_html(graph), encoding="utf-8")
    return page


async def capture(page_path: pathlib.Path) -> list[tuple[bytes, int]]:
    from playwright.async_api import async_playwright

    shots: list[tuple[bytes, int]] = []

    async def shot(hold: int) -> None:
        shots.append((await pg.screenshot(), hold))

    async with async_playwright() as p:
        # An installed-browser mismatch (a pinned Playwright version beside
        # a differently-numbered Chromium build) is the common way this
        # script fails on someone else's machine, and the stock error does
        # not mention that a path can be given. TOKENMIZER_CHROMIUM names
        # one; without it, Playwright's own lookup applies.
        exe = os.environ.get("TOKENMIZER_CHROMIUM", "")
        browser = await p.chromium.launch(
            **({"executable_path": exe} if exe else {}))
        pg = await browser.new_page(viewport=VIEWPORT)
        errors: list[str] = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        await pg.goto(page_path.as_uri())

        # Radial — the default, and the view the README leads with.
        await pg.wait_for_timeout(1800)
        await shot(_HOLD_LONG)

        # Click the busiest node: the detail panel is where a fact's
        # provenance lives, and it is the answer to "why should I trust
        # this?".
        await pg.wait_for_timeout(200)
        try:
            await pg.click("#hotList .hot", timeout=2000)
            # The detail card is further down the side panel than the row
            # that was clicked, so without this the frame shows the
            # selection highlighted in the graph and none of the
            # provenance that is the reason to select it.
            await pg.evaluate(
                "document.getElementById('detail')"
                ".scrollIntoView({block: 'center'})")
            # Selecting centres the view on the node, which crops the
            # radial circle; Fit is the control a reader would reach for
            # and it frames the whole graph with the selection still lit.
            await pg.click("#fit")
            await pg.wait_for_timeout(700)
            await shot(_HOLD_LONG)
            await pg.keyboard.press("Escape")
            # Leave the panel where the next beat expects it, or every
            # later frame inherits this one's scroll position.
            await pg.evaluate("document.getElementById('side').scrollTop = 0")
            await pg.wait_for_timeout(400)
        except Exception:
            pass  # hotspot list empty for this session — skip that beat

        # Force — clusters, the isolate column, settled before it paints.
        await pg.click("#viewGraph")
        await pg.wait_for_timeout(1500)
        await shot(_HOLD_LONG)

        # Filter a type out: the counts and the layout both respond.
        try:
            await pg.click("text=endpoint", timeout=2000)
            await pg.wait_for_timeout(900)
            await shot(_HOLD_SHORT)
            await pg.click("text=endpoint")
            await pg.wait_for_timeout(700)
        except Exception:
            pass

        # Timeline — the same session as a story.
        await pg.click("#viewTime")
        await pg.wait_for_timeout(1300)
        await shot(_HOLD_LONG)

        # Light theme, then back, so the loop returns where it started.
        await pg.click("#theme")
        await pg.wait_for_timeout(800)
        await shot(_HOLD_SHORT)
        await pg.click("#theme")
        await pg.click("#viewRadial")
        await pg.wait_for_timeout(1200)
        await shot(_HOLD_SHORT)

        await browser.close()

    if errors:
        raise SystemExit("The page raised errors; not shipping a GIF of a "
                         "broken page:\n  " + "\n  ".join(errors))
    return shots


def write_gif(shots: list[tuple[bytes, int]], out: pathlib.Path) -> None:
    import io

    from PIL import Image

    frames = []
    for raw, hold in shots:
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        # Rendered at exactly the width the README displays it at. Half
        # size was smaller but the browser then upscales it, and the node
        # labels — the whole reason to show the graph rather than describe
        # it — came out unreadable.
        if im.width != _TARGET_W:
            h = round(im.height * _TARGET_W / im.width)
            im = im.resize((_TARGET_W, h), Image.LANCZOS)
        # Adaptive palette per frame, then a shared one via the first
        # frame — GIF is 256 colours and the graph is mostly flat fills,
        # so this holds up well.
        frames.append((im.quantize(colors=200, method=Image.MEDIANCUT), hold))

    out.parent.mkdir(parents=True, exist_ok=True)
    first, rest = frames[0][0], [f for f, _ in frames[1:]]
    first.save(
        out, save_all=True, append_images=rest,
        duration=[hold for _, hold in frames], loop=0, optimize=True,
        disposal=2,
    )
    kb = out.stat().st_size / 1024
    print(f"{out}  {len(frames)} frames  {kb:.0f} KB")
    if kb > 5000:
        print("  warning: over 5 MB — GitHub will be slow to render this.")


def main() -> None:
    page = build_page()
    shots = asyncio.run(capture(page))
    if len(shots) < 3:
        raise SystemExit(f"Only {len(shots)} frames captured — the page "
                         f"controls may have changed.")
    write_gif(shots, OUT)


if __name__ == "__main__":
    main()
