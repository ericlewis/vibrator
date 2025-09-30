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


def rel_scale(values):
    mn = min(values)
    mx = max(values)
    if mx - mn <= 1e-12:
        return [0.5 for _ in values], mn, mx
    return [float((v - mn) / (mx - mn)) for v in values], mn, mx


def norm_01_from_cos(sim: float) -> float:
    # Map cosine-like [-1,1] to [0,1]
    return (float(sim) + 1.0) / 2.0


def softmax(xs: List[float], alpha: float = 8.0) -> List[float]:
    if not xs:
        return []
    m = max(xs)
    exps = [pow(2.718281828, (x - m) * alpha) for x in xs]
    s = sum(exps)
    return [e / s for e in exps] if s > 0 else [1.0 / len(xs)] * len(xs)


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


def run(text: str, items_file: Path | None, sliders_file: Path | None, contrast: bool = False, relative: bool = False, alpha: float = 8.0) -> None:
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
    print("What this tests: cosine-style similarity between your text and each slider description (no user profile).")
    if contrast:
        print("Includes relative bars normalized within this group to emphasize differences.")
    print()
    print(text)
    print()
    sorted_sims = sorted(zip(slider_names, sims.tolist()), key=lambda kv: kv[1], reverse=True)
    abs_vals = [norm_01_from_cos(sim) for _, sim in sorted_sims]
    rels, _, _ = rel_scale(abs_vals)
    rel_soft = softmax([sim for _, sim in sorted_sims], alpha=alpha) if relative else [None] * len(sorted_sims)
    for (name, sim), nv, rel, rs in zip(sorted_sims, abs_vals, rels, rel_soft):
        line = f"{name:>10}: {bar(nv)} {nv:.3f}"
        if relative and rs is not None:
            line += f"  | softmax {bar(rs, 12)} {int(round(rs*100)):>3}%"
        if contrast:
            line += f"  | rel {bar(rel, 12)} {int(round(rel*100)):>3}%"
        print(line)
    print()

    # 2) As an item: how would a baseline user score this text?
    scorer = SliderScorer(encoder, slider_vectors)
    scores = scorer.score(DEFAULT_ACTIONS, text, context={"segment": "work", "is_work_hours": True})
    print("Baseline User -> Item Probabilities")
    print("===================================")
    print("What this tests: pipeline probability per slider for a baseline demo user acting on your text.")
    print("(Weighted combination of user/slider, item/slider, user/item, and recency; sigmoid-calibrated.)")
    sorted_probs = sorted(scores.items(), key=lambda kv: kv[1].probability, reverse=True)
    abs_vals = [float(out.probability) for _, out in sorted_probs]
    rels, _, _ = rel_scale(abs_vals)
    for (name, out), rel in zip(sorted_probs, rels):
        p = float(out.probability)
        line = f"{name:>10}: {bar(p)} {p:.3f}"
        if relative and getattr(out, "relative_probability", None) is not None:
            rp = float(out.relative_probability)  # type: ignore[attr-defined]
            line += f"  | softmax {bar(rp, 12)} {int(round(rp*100)):>3}%"
        if contrast:
            line += f"  | rel {bar(rel, 12)} {int(round(rel*100)):>3}%"
        print(line)
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
            print("What this tests: nearest neighbors to your text among the provided items (by embedding similarity).")
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
    ap.add_argument("--contrast", action="store_true", help="Add relative bars normalized within the group")
    ap.add_argument("--relative", action="store_true", help="Show per-group softmax bars (sharper separation)")
    ap.add_argument("--alpha", type=float, default=8.0, help="Temperature for softmax separation on slider sims")
    args = ap.parse_args()

    if args.text == "@-":
        user_text = sys.stdin.read()
    else:
        user_text = args.text

    run(user_text.strip(), args.items_file, args.sliders_file, contrast=args.contrast, relative=args.relative, alpha=args.alpha)
