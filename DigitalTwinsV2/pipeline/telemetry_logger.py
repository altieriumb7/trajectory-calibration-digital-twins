"""Corrected telemetry logger. Drop-in replacement for
``phmforge_calibration/telemetry_logger.py``.

The governing rule: **this module never invents a value.** Where the original
filled a gap with a plausible number, this one records ``None`` and says so in a
per-step ``provenance`` block. A missing feature is a row you exclude, not a row
you synthesise.

Changes against the original, each tied to a defect:

=====================  ====================================================
original                corrected
=====================  ====================================================
``return 0.75``         ``None`` + ``provenance.confidence = "unavailable"``
``estimate_logprobs``   deleted; ``None`` + ``provenance.logprobs``
confidence once per     confidence parsed **per step**, so a confidence
trajectory              gradient is a measurement rather than a constant 0
``status_code`` from    ``status_code`` from the MCP response; ``None`` when
regex on observation    the call site did not supply one
observation holds       tool output in ``observation``; harness/grader text
grader verdicts         quarantined in ``harness_observation``
``execution_time_ms``   ``None`` unless actually timed
= 1200 fallback
=====================  ====================================================
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

# Actions handled by the agent framework itself, not by an MCP tool server.
NON_TOOL_ACTIONS = frozenset({"Finish", "Self-Ask", "Agent-Ask", ""})

# Verdict strings the harness writes once the outcome is known. Kept in sync with
# provenance.GRADER_PATTERNS.
GRADER_VERDICT = re.compile(
    r"\btask\s+(finished successfully|failed|succeeded|completed)\b", re.I
)

# A confidence report the model appended to its action input instead of its
# thought. It is parsed out as a feature, then removed from the recorded input:
# loop_ratio compares (action, action_input) pairs, so leaving a varying
# confidence in place would make two identical tool calls look distinct.
CONFIDENCE_TAIL = re.compile(r"\s*Confidence\s*[:=]\s*\d{1,3}\s*\.?\s*$", re.I)

# Ordered most- to least-specific. The bare-integer form is last and requires an
# explicit ':' or '=' separator, so ordinary prose ("confidence in 5 minutes")
# cannot match it. It is the form CONFIDENCE_PROMPT asks the agent to emit.
CONFIDENCE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"confidence\s*(?:level|score)?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%", re.I),
    re.compile(r"confidence\s*(?:level|score)?\s*[:=]?\s*(0\.\d+)\b", re.I),
    re.compile(r"(\d+(?:\.\d+)?)\s*%\s*confidence", re.I),
    re.compile(r"probability\s*of\s*success\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%", re.I),
    re.compile(r"confidence\s*(?:level|score)?\s*[:=]\s*(\d{1,3})\b", re.I),
)


class TelemetryLogger:
    """Captures a trajectory without imputing anything."""

    def __init__(self, output_dir: str | Path | None = None) -> None:
        # phmforge_runner constructs this with no arguments, so the destination
        # has to be overridable from outside to run a smoke test into a scratch
        # directory instead of the real corpus.
        if output_dir is None:
            output_dir = os.environ.get(
                "PHMFORGE_TELEMETRY_DIR", "results/telemetry_runs"
            )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Confidence
    # ------------------------------------------------------------------ #
    @staticmethod
    def parse_confidence(text: str | None) -> tuple[float | None, str]:
        """Parses a verbalized confidence out of ``text``.

        Returns ``(value, provenance)`` where provenance is ``"parsed"`` or
        ``"unavailable"``. There is deliberately no default: an unparseable
        step has no confidence, and the analysis must drop it.
        """
        if not text:
            return None, "unavailable"
        for pattern in CONFIDENCE_PATTERNS:
            match = pattern.search(text)
            if not match:
                continue
            value = float(match.group(1))
            if value > 1.0:
                value /= 100.0
            if 0.0 <= value <= 1.0:
                return value, "parsed"
        return None, "unavailable"

    # ------------------------------------------------------------------ #
    # Observations
    # ------------------------------------------------------------------ #
    @staticmethod
    def split_observation(
        step: dict[str, Any], is_terminal: bool
    ) -> tuple[str | None, str | None]:
        """Separates tool output from harness-written text.

        The grader's verdict ("Task finished successfully" / "Task failed") is
        written after the outcome is known. Returning it as ``observation``
        lets any text-derived feature read the label, which is what produced
        V1's apparent AUROC of 0.881. It goes to a quarantined field instead.
        """
        raw = str(step.get("observation") or step.get("raw_observation_output") or "")
        # Quarantine on evidence, not on position. A trajectory that stops by
        # exhausting its step budget has ordinary tool output in its last
        # observation, and discarding that would throw away a real measurement.
        # Only a Finish action or an explicit verdict string is harness-written.
        harness_written = step.get("action", "") == "Finish" or bool(
            GRADER_VERDICT.search(raw)
        )
        return (None, raw) if harness_written else (raw, None)

    # ------------------------------------------------------------------ #
    # Tool calls
    # ------------------------------------------------------------------ #
    @staticmethod
    def build_tool_calls(step: dict[str, Any]) -> list[dict[str, Any]]:
        """Builds MCP call records from *response metadata only*.

        The original inferred ``status_code`` by string-matching the
        observation for "Error"/"failed", which makes ``mcp_error_ratio`` a
        text heuristic rather than telemetry. Here the status must be supplied
        by the call site; otherwise it is ``None`` and the row is unusable for
        error-rate features.
        """
        action = step.get("action", "")
        if action in NON_TOOL_ACTIONS:
            return []

        raw_calls = step.get("mcp_tool_calls")
        if raw_calls:
            out = []
            for call in raw_calls:
                status = call.get("status_code")
                out.append(
                    {
                        "tool_name": call.get("tool_name", action),
                        "payload": call.get("payload", step.get("action_input", "")),
                        "status_code": status,
                        "execution_time_ms": call.get("execution_time_ms"),
                        "source": "mcp_response" if status is not None else "unavailable",
                    }
                )
            return out

        return [
            {
                "tool_name": action,
                "payload": step.get("action_input", ""),
                "status_code": step.get("status_code"),
                "execution_time_ms": step.get("execution_time_ms"),
                "source": "mcp_response" if step.get("status_code") is not None else "unavailable",
            }
        ]

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #
    def log_trajectory(
        self,
        model_name: str,
        scenario_id: str,
        steps_log: Iterable[dict[str, Any]],
        task_success: bool,
        final_answer: str | None = None,
        raw_scratchpad: str | None = None,
    ) -> Path:
        """Writes one trajectory as JSONL, one record per step.

        ``raw_scratchpad`` is accepted for call-compatibility with
        ``phmforge_runner.run_one_scenario``, which passes it positionally by
        keyword. It is used only as a last-resort source for the final
        confidence, never to fill per-step fields. Keeping the parameter matters:
        the runner wraps this call in a bare ``except`` that degrades a
        signature mismatch into a printed warning, so an incompatible signature
        would silently produce no telemetry at all for a paid run.
        """
        safe_model = model_name.replace("/", "_").replace(":", "_")
        model_dir = self.output_dir / safe_model
        model_dir.mkdir(parents=True, exist_ok=True)
        output_file = model_dir / f"{scenario_id}.jsonl"

        steps = list(steps_log)
        with output_file.open("w", encoding="utf-8") as handle:
            for idx, step in enumerate(steps):
                is_terminal = idx == len(steps) - 1

                # Logprobs: provider-supplied or absent. Never generated.
                logprobs = step.get("thought_logprobs")
                logprob_provenance = "provider" if logprobs else "unavailable"

                # Confidence: per step, so a gradient is meaningful.
                # Models frequently emit the confidence line after the action
                # input rather than inside the thought, so both are scanned.
                confidence, conf_provenance = self.parse_confidence(step.get("thought"))
                if confidence is None:
                    confidence, conf_provenance = self.parse_confidence(
                        step.get("action_input")
                    )
                if confidence is None and is_terminal:
                    for fallback in (final_answer, raw_scratchpad):
                        if not fallback:
                            continue
                        confidence, conf_provenance = self.parse_confidence(fallback)
                        if confidence is not None:
                            break

                observation, harness_observation = self.split_observation(step, is_terminal)
                tool_calls = self.build_tool_calls(step)

                # Confidence has already been parsed above; remove its textual
                # trace so loop detection compares actual tool arguments.
                action_input = step.get("action_input", "")
                if isinstance(action_input, str):
                    action_input = CONFIDENCE_TAIL.sub("", action_input).rstrip()

                record = {
                    "step_index": idx,
                    "thought": step.get("thought", ""),
                    "action": step.get("action", ""),
                    "action_input": action_input,
                    "observation": observation,
                    "harness_observation": harness_observation,
                    "thought_logprobs": logprobs or None,
                    "verbalized_confidence": confidence,
                    "mcp_tool_calls": tool_calls,
                    "provenance": {
                        "logprobs": logprob_provenance,
                        "confidence": conf_provenance,
                        "status_code": (
                            "mcp_response"
                            if tool_calls and all(c["status_code"] is not None for c in tool_calls)
                            else "unavailable"
                        ),
                    },
                }
                if is_terminal:
                    record["task_success"] = bool(task_success)
                handle.write(json.dumps(record) + "\n")

        return output_file
