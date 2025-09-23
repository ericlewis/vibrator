"""Align input text against sliders and optionally an items file.

Examples:
  uv run python examples/align_input.py --text "Async database tuning with Postgres"
  uv run python examples/align_input.py --text @- < input.txt
  uv run python examples/align_input.py --text "Karaoke lava surfing" --items-file examples/items.txt
  VIBRATOR_NOW_ISO=2024-01-01T12:00:00Z uv run python examples/align_input.py --text "..."

--items-file can be:
  - .txt (one item per line)
  - .json with {"items": ["..."]}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np

from vibrator import InstructionalEncoder, SliderScorer, UserAction
from vibrator.utils import seed_everything, stable_now, friendly_name


SLIDERS: Dict[str, str] = {
    "technical": "Detailed technical documentation and precise language.",
    "beginner": "Simple, step-by-step explanations and gentle ramp-up.",
    "creative": "Playful, imaginative, and unconventional tone.",
}

DEFAULT_ACTIONS = [
    UserAction("write", "Seeking help planning roadmap and code review cadence", weight=1.3),
    UserAction("click", "Loved a post about debugging asyncio race conditions", weight=1.0),
    UserAction("react", "Upvoted a playful analogy about threads as cats", weight=0.8),
]


def bar(value: float, width: int = 24) -> str:
    value = max(0.0, min(1.0, float(value)))
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)


def norm_01_from_cos(sim: float) -> float:
    # Map cosine-like [-1,1] to [0,1]
    return (float(sim) + 1.0) / 2.0


def load_items(path: Path) -> List[str]:
    txt = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(txt)
        items = data.get("items")
        if isinstance(items, list):
            return [str(x) for x in items if str(x).strip()]
        return []
    # default: treat as lines
    return [line.strip() for line in txt.splitlines() if line.strip()]


def load_sliders(path: Path | None) -> Dict[str, str]:
    if not path:
        return dict(SLIDERS)
    data = json.loads(path.read_text(encoding="utf-8"))
    sliders = data.get("sliders") if isinstance(data, dict) else None
    if not isinstance(sliders, dict):
        raise ValueError("--sliders-file must contain a JSON object with a 'sliders' mapping")
    return {str(k): str(v) for k, v in sliders.items()}


def run(text: str, items_file: Path | None, sliders_file: Path | None) -> None:
    seed_everything(0)
    _ = stable_now()  # anchor

    encoder = InstructionalEncoder()
    sliders = load_sliders(sliders_file)
    slider_vectors = dict(zip(sliders, encoder.encode_items(sliders.values())))

    # 1) Vibe alignment of input against sliders
    item_vec = encoder.encode_items([text])[0]
    slider_names = list(slider_vectors.keys())
    slider_mat = np.stack([slider_vectors[n] for n in slider_names], axis=0)
    sims = slider_mat @ item_vec

    print("Input Vibe Alignment")
    print("=====================")
    print(text)
    print()
    for name, sim in sorted(zip(slider_names, sims.tolist()), key=lambda kv: kv[1], reverse=True):
        nv = norm_01_from_cos(sim)
        print(f"{name:>10}: {bar(nv)} {nv:.3f}")
    print()

    # 2) As an item: how would a baseline user score this text?
    scorer = SliderScorer(encoder, slider_vectors)
    scores = scorer.score(DEFAULT_ACTIONS, text, context={"segment": "work", "is_work_hours": True})
    print("Baseline User -> Item Probabilities")
    print("===================================")
    for name, out in sorted(scores.items(), key=lambda kv: kv[1].probability, reverse=True):
        p = float(out.probability)
        print(f"{name:>10}: {bar(p)} {p:.3f}")
    print()

    # 3) Optional: top-5 similar items from a file
    if items_file and items_file.exists():
        items = load_items(items_file)
        if items:
            item_vecs = encoder.encode_items(items)
            sims_items = item_vecs @ item_vec
            order = np.argsort(-sims_items)[:5]
            print("Top-5 similar items from file")
            print("==============================")
            for idx in order.tolist():
                sim = float(sims_items[idx])
                nv = norm_01_from_cos(sim)
                print(f"[{nv:.3f}] {items[idx]}")
            print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="Input text, or @- to read stdin")
    ap.add_argument("--items-file", type=Path, default=None, help="Optional path to .txt/.json items file")
    ap.add_argument("--sliders-file", type=Path, default=None, help="Optional path to sliders.json with {\"sliders\": {...}}")
    args = ap.parse_args()

    if args.text == "@-":
        user_text = sys.stdin.read()
    else:
        user_text = args.text

    run(user_text.strip(), args.items_file, args.sliders_file)
