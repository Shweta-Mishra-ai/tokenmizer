# -*- coding: utf-8 -*-
"""Generate docs/assets/demo.gif — terminal-style demo.

All numbers/outputs shown are REAL, captured from an actual run of
GraphMemory + CheckpointManager on the 40-turn-scale demo session
(checkpoint ckpt_21a0959c3ddf, 25 nodes, 233-token resume).
Run: python scripts/gen_demo_gif.py [output_path]
"""
import os
import sys

from PIL import Image, ImageDraw, ImageFont

OUT = sys.argv[1] if len(sys.argv) > 1 else "docs/assets/demo.gif"
W, H = 920, 500
BG, FG = (13, 17, 23), (201, 209, 217)
GREEN, PURPLE, YELLOW = (63, 185, 80), (137, 87, 229), (210, 153, 34)
DIM, CYAN = (110, 118, 129), (57, 197, 187)

font = ImageFont.truetype("consola.ttf", 17)
bold = ImageFont.truetype("consolab.ttf", 17)
big = ImageFont.truetype("consolab.ttf", 26)


def frame(lines, title="tokenmizer - demo"):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([8, 8, W - 8, H - 8], 10, outline=(48, 54, 61), width=2)
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([24 + i * 24, 20, 38 + i * 24, 34], fill=c)
    d.text((W // 2 - 100, 18), title, font=font, fill=DIM)
    y = 56
    for txt, col, f in lines:
        d.text((28, y), txt, font=f, fill=col)
        y += 24
    return im


frames, durs = [], []


def add(lines, ms):
    frames.append(frame(lines))
    durs.append(ms)


add([("$ pip install tokenmizer", FG, bold),
     ("$ tokenmizer serve", FG, bold),
     ("  Proxy: http://localhost:8000/v1  (point any OpenAI client here)", CYAN, font),
     ("", FG, font),
     ("session: fastapi-auth", DIM, font),
     ("you > Build a FastAPI auth service with JWT + PostgreSQL", FG, font),
     ("you > Use bcrypt / fix the 422 / add Redis refresh tokens ...", FG, font),
     ("            ... 40 turns later ...", YELLOW, bold),
     ("", FG, font),
     ("  context window: 87% full", YELLOW, bold)], 2400)

add([("  context window: 87% full", YELLOW, bold),
     ("", FG, font),
     ("  * auto-checkpoint triggered *", PURPLE, bold),
     ("", FG, font),
     ("  graph: 25 nodes (tasks / decisions / files / errors)", FG, font),
     ("  checkpoint: ckpt_21a0959c3ddf   saved to SQLite", GREEN, font),
     ("", FG, font),
     ("  ~5,800 tokens of history  ->  233 tokens", CYAN, bold)], 2400)

add([("  -- next day, fresh session --", DIM, font),
     ("", FG, font),
     ("$ tokenmizer resume fastapi-auth", FG, bold),
     ("", FG, font),
     ("Goal: a FastAPI auth service with JWT and PostgreSQL", GREEN, font),
     ("Working on: rate limiting with slowapi", FG, font),
     ("Done: tests/test_auth.py (12 tests passing) | 422 login fix", FG, font),
     ("Decided: PostgreSQL | bcrypt | Python 3.12 | Redis refresh tokens", FG, font),
     ("Changes: 'Use Redis' -> 'Redis for refresh token storage'", YELLOW, font),
     ("Continue from: Write the tests", CYAN, font),
     ("", FG, font),
     ("  [233 tokens - paste anywhere, keep going]", DIM, font)], 3400)

im = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(im)
d.rounded_rectangle([8, 8, W - 8, H - 8], 10, outline=(48, 54, 61), width=2)
d.text((W // 2 - 260, H // 2 - 70), "never re-explain your project again", font=big, fill=FG)
d.text((W // 2 - 215, H // 2 - 20), "~5,800 tokens  ->  233 tokens resume", font=big, fill=GREEN)
d.text((W // 2 - 130, H // 2 + 40), "pip install tokenmizer", font=bold, fill=PURPLE)
frames.append(im)
durs.append(2600)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
frames[0].save(OUT, save_all=True, append_images=frames[1:],
               duration=durs, loop=0, optimize=True)
print("demo.gif:", os.path.getsize(OUT) // 1024, "KB,", len(frames), "frames")
