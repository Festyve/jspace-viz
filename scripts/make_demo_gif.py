# Copyright 2026 Michael Zhang
# SPDX-License-Identifier: Apache-2.0
"""Assemble the README demo GIF from real renders of a running server.

Captures the app reading growing prefixes of a prompt (each frame is a real
headless-Chrome render of a real read — nothing mocked), then assembles them
into assets/demo.gif.

    jspace-viz --preset deepseek-coder-1.3b &   # server on :8321
    python scripts/make_demo_gif.py
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import tempfile
import urllib.parse

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PROMPT = "nums = [3, 1, 2]\nnums.sort()\nprint(nums[-1])\n# This prints"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8321/")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--out", default="assets/demo.gif")
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args()

    # One prefix per word, whitespace (incl. newlines) preserved.
    parts = re.split(r"(\s+)", args.prompt)
    prefixes: list[str] = []
    acc = ""
    for part in parts:
        acc += part
        if part.strip():
            prefixes.append(acc)
    prefixes = prefixes[2:]  # skip 1-2 word prefixes (not much to see)

    frames: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        # opening frame: the blank-slate welcome screen
        shots = [("welcome", args.url)] + [
            (f"step{i:02d}", f"{args.url}?prompt={urllib.parse.quote(p)}")
            for i, p in enumerate(prefixes)
        ]
        for name, url in shots:
            path = os.path.join(tmp, f"{name}.png")
            # Chrome's virtual-time screenshot mode occasionally deadlocks on
            # slow in-page fetches — time out and retry once.
            for attempt in (1, 2):
                try:
                    subprocess.run(
                        [
                            CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                            f"--screenshot={path}",
                            f"--window-size={args.width},{args.height}",
                            "--virtual-time-budget=90000",
                            url,
                        ],
                        check=True,
                        capture_output=True,
                        timeout=150,
                    )
                    break
                except subprocess.TimeoutExpired:
                    print(f"{name}: attempt {attempt} timed out", flush=True)
            else:
                raise RuntimeError(f"could not capture {name}")
            frames.append(path)
            print(f"captured {name}", flush=True)

        from PIL import Image

        images = [Image.open(f).convert("RGB").quantize(colors=128) for f in frames]
        durations = [1400] + [700] * (len(images) - 2) + [3500]
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        images[0].save(
            args.out,
            save_all=True,
            append_images=images[1:],
            duration=durations,
            loop=0,
            optimize=True,
        )
    print(f"wrote {args.out} ({os.path.getsize(args.out) // 1024} KB, {len(images)} frames)")


if __name__ == "__main__":
    main()
