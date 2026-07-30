"""Corrected feature extractor. Drop-in replacement for
``phmforge_calibration/feature_extractor.py``.

Two rules the original broke:

1. **Never read quarantined fields.** ``harness_observation`` holds the grader's
   verdict and is invisible here. Only ``observation`` (tool output) is used.
2. **Never compute a feature from unmeasured data.** If a step's provenance says
   a signal was not captured, the feature is ``NaN`` and the trajectory is
   marked unusable for that family, rather than silently imputed.

Features are grouped by family so a caller can ask "which trajectories can
support a token-uncertainty analysis?" and get an honest answer.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Families are split by *provenance*, not by topic. Trace-shape features are
# always computable; error-rate features need tool status codes the agent
# framework may not supply. Lumping them together would discard usable
# structural signal whenever status codes are missing.
FAMILIES: dict[str, tuple[str, ...]] = {
    "token": ("early_entropy", "logprob_variance", "mean_logprob"),
    "verbalized": ("verbalized_final", "confidence_gradient"),
    "structural": (
        "n_steps",
        "n_tool_calls",
        "loop_ratio",
        "n_unique_actions",
        "reached_finish",
    ),
    "mcp_error": ("mcp_error_ratio", "mcp_error_count"),
    "observation": ("obs_mean_len", "obs_max_len", "n_observations"),
}

ALL_FEATURES: tuple[str, ...] = tuple(f for fam in FAMILIES.values() for f in fam)


def _slope(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    return float(np.polyfit(np.arange(len(values)), values, 1)[0])


class FeatureExtractor:
    """Builds a trajectory-level feature table with provenance gating."""

    def extract_trajectory(self, path: Path) -> dict[str, Any]:
        steps = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not steps:
            return {}

        # --- provenance gates ------------------------------------------------
        provs = [s.get("provenance", {}) or {} for s in steps]
        token_ok = any(p.get("logprobs") == "provider" for p in provs)
        conf_steps = [
            s.get("verbalized_confidence")
            for s, p in zip(steps, provs)
            if p.get("confidence") == "parsed" and s.get("verbalized_confidence") is not None
        ]
        verbalized_ok = len(conf_steps) > 0
        calls = [c for s in steps for c in (s.get("mcp_tool_calls") or [])]
        status_ok = bool(calls) and all(c.get("status_code") is not None for c in calls)

        # --- token features (gated) ------------------------------------------
        all_lp = [lp for s in steps for lp in (s.get("thought_logprobs") or [])]
        early_lp = [lp for s in steps[:2] for lp in (s.get("thought_logprobs") or [])]
        if token_ok and all_lp:
            early_entropy = -float(np.mean(early_lp)) if early_lp else float("nan")
            logprob_variance = float(np.var(all_lp))
            mean_logprob = float(np.mean(all_lp))
        else:
            early_entropy = logprob_variance = mean_logprob = float("nan")

        # --- verbalized features (gated) -------------------------------------
        if verbalized_ok:
            verbalized_final = float(conf_steps[-1])
            confidence_gradient = _slope([float(c) for c in conf_steps])
        else:
            verbalized_final = confidence_gradient = float("nan")

        # --- execution features ----------------------------------------------
        actions = [s.get("action", "") for s in steps if s.get("action")]
        inputs = [s.get("action_input", "") for s in steps if s.get("action_input")]
        n_err = sum(1 for c in calls if c.get("status_code") not in (None, 200))
        mcp_error_ratio = (n_err / len(calls)) if (status_ok and calls) else float("nan")
        mcp_error_count = float(n_err) if status_ok else float("nan")

        # --- observation features (tool output only) --------------------------
        # harness_observation is never read.
        obs_lens = [
            len(str(s["observation"]))
            for s in steps
            if s.get("observation") is not None
        ]

        return {
            "scenario": path.stem,
            "model": path.parent.name,
            # token
            "early_entropy": early_entropy,
            "logprob_variance": logprob_variance,
            "mean_logprob": mean_logprob,
            # verbalized
            "verbalized_final": verbalized_final,
            "confidence_gradient": confidence_gradient,
            # execution
            "n_steps": float(len(steps)),
            "n_tool_calls": float(len(calls)),
            "mcp_error_ratio": mcp_error_ratio,
            "mcp_error_count": mcp_error_count,
            "loop_ratio": (
                (len(actions) - len(set(zip(actions, inputs)))) / len(actions)
                if actions
                else float("nan")
            ),
            "n_unique_actions": float(len(set(actions))),
            "reached_finish": float(any(a == "Finish" for a in actions)),
            # observation
            "obs_mean_len": float(np.mean(obs_lens)) if obs_lens else float("nan"),
            "obs_max_len": float(np.max(obs_lens)) if obs_lens else float("nan"),
            "n_observations": float(len(obs_lens)),
            # gates. Structural features derive from the trace shape and need no
            # provider or protocol support, so they are always available.
            "usable_token": token_ok,
            "usable_verbalized": verbalized_ok,
            "usable_structural": True,
            "usable_mcp_error": status_ok,
            "usable_observation": len(obs_lens) > 0,
            # label
            "y": int(bool(steps[-1].get("task_success", False))),
        }

    def extract_dataset(self, telemetry_dir: Path) -> pd.DataFrame:
        rows = [
            row
            for path in sorted(Path(telemetry_dir).rglob("*.jsonl"))
            if (row := self.extract_trajectory(path))
        ]
        return pd.DataFrame(rows)

    @staticmethod
    def usable(df: pd.DataFrame, family: str) -> pd.DataFrame:
        """Rows that genuinely support ``family``, with a printed accounting."""
        gate = f"usable_{family}"
        if gate not in df.columns:
            return df
        keep = df[df[gate]]
        dropped = len(df) - len(keep)
        if dropped:
            print(
                f"  [{family}] {len(keep)}/{len(df)} trajectories usable "
                f"({dropped} dropped: signal not measured)"
            )
        return keep
