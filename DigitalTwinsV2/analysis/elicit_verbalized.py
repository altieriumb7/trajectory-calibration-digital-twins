#!/usr/bin/env python3
"""Post-hoc verbalized confidence, from the recorded reasoning text.

The verbalized family is the standard comparison for any token-uncertainty
claim, and it is missing from our corpus: eliciting confidence in-band made
models over-generate past the action input, which corrupted the tool argument
and drove the agent into single-action loops (Pass@1 fell to 0). See AUDIT.md.

Eliciting it afterwards avoids that entirely: nothing is re-run and the agent is
never perturbed, so the measurement cannot change the thing measured.

Each trajectory is judged by **the model that produced it**. That matters. A
third-party judge reading the text would measure how confident the reasoning
*sounds*, which is a different quantity from what the literature means by
verbalized confidence -- there, the model reports on its own belief
(Kadavath et al.; Xiong et al.). Asking the original model keeps it self-report;
only the timing changes, from in-band to post-hoc.

The residual gap from the literature is that the model no longer holds the
generation state it had at the time, so this is self-report from re-reading
rather than introspection during decoding. The paper should say so.

Usage:  python analysis/elicit_verbalized.py results/v2_clean
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_V2 = Path(__file__).resolve().parent.parent
_ROOT = _V2.parent
sys.path.insert(0, str(_V2))

try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env", override=False)
except ImportError:
    pass

BASE_URL = "https://api.together.xyz/v1"

# Directory name -> the provider model id that produced those trajectories, so
# each one is asked about its own work rather than judged by a stand-in.
DIR_TO_MODEL = {
    "openai_meta-llama_Llama-3.3-70B-Instruct-Turbo":
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "openai_Qwen_Qwen2.5-7B-Instruct-Turbo":
        "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "openai_openai_gpt-oss-120b": "openai/gpt-oss-120b",
}

PROMPT = (
    "Below is reasoning you produced earlier while attempting an industrial "
    "prognostics task.\n\n"
    "How confident are you that this approach solved the task correctly? Report "
    "your own belief. You are not told the outcome, so do not guess it -- judge "
    "only your confidence in the approach as you pursued it.\n\n"
    "Answer with a single integer from 0 to 100 and nothing else.\n\n"
    "--- YOUR REASONING ---\n{body}\n--- END ---\n\nConfidence:"
)

MAX_CHARS = 6000


def load_thoughts(path: Path) -> str:
    steps = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    body = "\n".join(str(s.get("thought") or "").strip() for s in steps)
    body = "\n".join(l for l in body.splitlines() if l)
    return body[:MAX_CHARS]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--out", type=Path,
                    default=_V2 / "tables" / "verbalized_posthoc.csv")
    args = ap.parse_args()

    key = os.environ.get("TOGETHER_API_KEY", "")
    if not key:
        print("set TOGETHER_API_KEY in .env")
        return 1

    from openai import OpenAI

    client = OpenAI(api_key=key, base_url=BASE_URL)
    paths = sorted(args.corpus.rglob("*.jsonl"))
    print(f"eliciting self-reported confidence on {len(paths)} trajectories")

    rows, skipped, failed = [], 0, 0
    for i, path in enumerate(paths, 1):
        body = load_thoughts(path)
        if len(body) < 40:
            skipped += 1
            continue
        model_id = DIR_TO_MODEL.get(path.parent.name)
        if model_id is None:
            skipped += 1
            continue
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": PROMPT.format(body=body)}],
                max_tokens=64, temperature=0)
            text = (resp.choices[0].message.content or "").strip()
            digits = "".join(c for c in text[:8] if c.isdigit())
            if not digits:
                failed += 1
                continue
            value = max(0, min(100, int(digits))) / 100.0
        except Exception as exc:  # noqa: BLE001
            failed += 1
            if failed <= 3:
                print(f"  [err] {path.stem}: {type(exc).__name__} {str(exc)[:70]}")
            time.sleep(2)
            continue
        rows.append({"model": path.parent.name, "scenario": path.stem,
                     "verbalized_posthoc": value})
        if i % 25 == 0:
            print(f"  {i}/{len(paths)}")

    import pandas as pd

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n  judged {len(df)}  skipped {skipped} (no reasoning)  failed {failed}")
    if not df.empty:
        v = df.verbalized_posthoc
        print(f"  distribution: mean={v.mean():.3f} std={v.std():.3f} "
              f"distinct={v.nunique()}")
        if v.nunique() < 3:
            print("  WARNING: near-constant -- the judge is not discriminating, "
                  "so this proxy carries no information and must not be reported "
                  "as a baseline.")
    print(f"  written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
