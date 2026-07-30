"""Provenance validation for agentic-trajectory telemetry.

This is the reusable core of the V2 correction. It answers one question before
any modelling happens: *is every feature column actually a measurement?*

Four checks, each targeting a defect found in the V1 corpus:

``constant``    a column with <2 distinct values carries no information, and any
                importance attributed to it is noise (V1: verbalized confidence).
``synthetic``   token logprobs drawn from a fallback generator instead of a
                provider. Real logprobs pile up near 0 with a long left tail;
                a fitted-Gaussian fit that *succeeds* is the tell (V1: all
                203,107 tokens).
``leakage``     grader or harness verdict strings inside fields that feature
                extraction can reach (V1: "Task failed" in ``observation``).
``imputed``     rows whose provenance flag says the value was never measured.

Findings are reported per column, and columns map to feature families. A defect
disqualifies the families that depend on it, not the whole dataset: the CLI exits
non-zero only when no measured family survives. That distinction is the point --
"drop this feature" and "discard this corpus" are different instructions, and a
gate that conflates them gets switched off.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

# Verdict strings written by the grading harness *about the trajectory itself*,
# after the outcome is known. Kept deliberately narrow: a gate that cries wolf
# gets switched off. Domain phrases like "ground truth RUL values" are ordinary
# tool output and must not match.
GRADER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\btask\s+(finished successfully|failed|succeeded|completed)\b", re.I),
    re.compile(r"\b(the\s+)?(answer|prediction)\s+(is\s+)?(in)?correct\b", re.I),
    re.compile(r"\b(evaluation|grader)\s+(verdict|result)\s*[:=]", re.I),
    re.compile(r"\btask_success\s*[:=]\s*(true|false|[01])\b", re.I),
)

# A provenance flag of anything in this set means "not measured".
UNMEASURED = {"unavailable", "imputed", "synthetic", "default", None}

# Which columns each feature family depends on. "structural" (step counts, loop
# ratio, action diversity) is derived from the trace shape and always available.
FAMILY_COLUMNS: dict[str, tuple[str, ...]] = {
    "token": ("thought_logprobs",),
    "verbalized": ("verbalized_confidence",),
    "mcp_error": ("mcp_status_code",),
    "structural": (),
}

# Provenance flag name -> the feature family that depends on it, so an
# availability leak can disqualify the right family.
PROVENANCE_TO_FAMILY: dict[str, str] = {
    "logprobs": "token",
    "confidence": "verbalized",
    "status_code": "mcp_error",
}


@dataclass
class Finding:
    """One provenance problem."""

    check: str
    column: str
    severity: str  # "fatal" | "warn"
    detail: str

    def __str__(self) -> str:
        mark = "FATAL" if self.severity == "fatal" else "warn "
        return f"  [{mark}] {self.check}/{self.column}: {self.detail}"


@dataclass
class Report:
    n_trajectories: int = 0
    n_steps: int = 0
    findings: list[Finding] = field(default_factory=list)

    @property
    def fatal(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "fatal"]

    @property
    def ok(self) -> bool:
        return not self.fatal

    def add(self, check: str, column: str, severity: str, detail: str) -> None:
        self.findings.append(Finding(check, column, severity, detail))

    def usable_families(self) -> tuple[list[str], list[str]]:
        """Splits feature families into usable and unusable.

        A column-level defect disqualifies the families that depend on it, not
        the whole corpus. Reporting that distinction is the difference between
        'drop this feature' and 'discard this dataset', and conflating them
        would make the gate useless in exactly the case it is meant for.
        """
        broken = {f.column for f in self.fatal}
        # An availability leak disqualifies the family that carries the column,
        # even though every recorded value in it may be genuine.
        leaked = {PROVENANCE_TO_FAMILY[f.column]
                  for f in self.fatal
                  if f.check == "availability-leak" and f.column in PROVENANCE_TO_FAMILY}
        ok, bad = [], []
        for family, columns in FAMILY_COLUMNS.items():
            unusable = family in leaked or any(c in broken for c in columns)
            (bad if unusable else ok).append(family)
        return ok, bad

    @property
    def corpus_unusable(self) -> bool:
        """True when no family survives, the corpus is empty, or the runs failed.

        A degenerate corpus is rejected outright: if the trajectories record no
        agent behaviour, no feature family is meaningful regardless of how
        clean its provenance looks.
        """
        if any(f.column == "*" for f in self.fatal):
            return True
        if any(f.check == "degenerate" for f in self.fatal):
            return True
        usable, _ = self.usable_families()
        return usable == ["structural"] or not usable

    def render(self) -> str:
        lines = [
            "=" * 74,
            f"PROVENANCE REPORT  ({self.n_trajectories} trajectories, {self.n_steps} steps)",
            "=" * 74,
        ]
        if not self.findings:
            lines.append("  no findings -- every column is a measurement")
        else:
            lines.extend(str(f) for f in self.findings)
        lines.append("-" * 74)

        usable, unusable = self.usable_families()
        lines.append(f"  usable feature families:   {', '.join(usable) or 'none'}")
        if unusable:
            lines.append(f"  unusable feature families: {', '.join(unusable)}")

        if self.corpus_unusable:
            lines.append(
                f"VERDICT: REJECTED -- {len(self.fatal)} fatal finding(s); "
                "no measured signal family survives"
            )
        elif self.fatal:
            lines.append(
                "VERDICT: usable with exclusions -- train only on the families "
                "listed as usable"
            )
        else:
            lines.append("VERDICT: corpus is usable")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Statistical helpers
# --------------------------------------------------------------------------- #
def _skewness(x: np.ndarray) -> float:
    if len(x) < 3 or x.std() == 0:
        return 0.0
    return float(np.mean(((x - x.mean()) / x.std()) ** 3))


def _ks_vs_normal(x: np.ndarray) -> float:
    """Two-sided KS statistic of ``x`` against the normal it best fits.

    Implemented directly to avoid a scipy dependency. A *small* statistic means
    the sample looks Gaussian -- which for token logprobs is the alarm, not the
    reassurance.
    """
    if len(x) < 8:
        return 1.0
    mu, sigma = float(x.mean()), float(x.std(ddof=1))
    if sigma == 0:
        return 1.0
    xs = np.sort(x)
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)((xs - mu) / (sigma * math.sqrt(2.0))))
    n = len(xs)
    upper = np.arange(1, n + 1) / n - cdf
    lower = cdf - np.arange(0, n) / n
    return float(max(upper.max(), lower.max()))


def looks_synthetic(logprobs: Sequence[float]) -> tuple[bool, str]:
    """Heuristic detector for fabricated token logprobs.

    Genuine logprobs are dominated by near-certain tokens, so they concentrate
    just below 0 with a long left tail: strongly negative skew, poor Gaussian
    fit, and a sizeable mass above -0.01. A fallback generator that samples
    ``N(mu, sigma)`` produces the opposite on all three.
    """
    x = np.asarray([v for v in logprobs if v is not None], dtype=float)
    if len(x) < 50:
        return False, "too few tokens to judge"

    skew = _skewness(x)
    ks = _ks_vs_normal(x)
    near_certain = float(np.mean(x > -0.01))

    gaussian_like = ks < 0.05
    symmetric = abs(skew) < 0.75
    no_certain_mass = near_certain < 0.02

    votes = sum((gaussian_like, symmetric, no_certain_mass))
    detail = (
        f"skew={skew:+.2f} (real: strongly negative), "
        f"KS-vs-normal={ks:.3f} (real: >0.05), "
        f"mass>-0.01={near_certain:.1%} (real: >2%)"
    )
    return votes >= 2, detail


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def check_constant(
    report: Report,
    column: str,
    values: Sequence[Any],
    warn_low_cardinality: bool = True,
) -> None:
    """Flags columns that cannot carry information.

    ``warn_low_cardinality`` is disabled for genuinely categorical columns such
    as HTTP status codes, where a handful of distinct values is healthy rather
    than suspicious.
    """
    vals = [v for v in values if v is not None]
    if not vals:
        report.add("constant", column, "fatal", "every value is null")
        return
    distinct = sorted({v for v in vals})
    if len(distinct) < 2:
        report.add(
            "constant", column, "fatal",
            f"single distinct value {distinct[0]!r} across {len(vals)} rows -- "
            "carries no information; any attributed importance is noise",
        )
    elif warn_low_cardinality and len(distinct) <= 3 and len(vals) > 50:
        report.add(
            "constant", column, "warn",
            f"only {len(distinct)} distinct values {distinct} across {len(vals)} rows",
        )


def check_synthetic(report: Report, column: str, logprobs: Sequence[float]) -> None:
    synthetic, detail = looks_synthetic(logprobs)
    if synthetic:
        report.add(
            "synthetic", column, "fatal",
            f"distribution matches a fallback generator, not a provider -- {detail}",
        )


def check_leakage(report: Report, column: str, texts: Iterable[str]) -> None:
    hits: list[str] = []
    total = 0
    for t in texts:
        total += 1
        for pat in GRADER_PATTERNS:
            m = pat.search(t or "")
            if m:
                hits.append(m.group(0))
                break
    if hits:
        sample = sorted({h.lower() for h in hits})[:3]
        report.add(
            "leakage", column, "fatal",
            f"{len(hits)}/{total} values contain harness/grader verdict text "
            f"(e.g. {sample}) -- move these to a quarantined field",
        )


def check_degenerate(report: Report, trajectories: Sequence[dict[str, Any]]) -> None:
    """Flags trajectories that record no agent behaviour at all.

    A run whose steps carry no extracted action is not a hard case -- it is a
    collection failure. The provider refused, the parse produced nothing, the
    process died. Every value in it is individually well-formed, so no
    value-level check objects, and the rows sail into the analysis as if the
    agent had simply performed badly.
    """
    if not trajectories:
        return
    empty = [t for t in trajectories if not t["n_actions"]]
    if not empty:
        return
    frac = len(empty) / len(trajectories)
    severity = "fatal" if frac > 0.10 else "warn"
    report.add(
        "degenerate", "trajectory", severity,
        f"{len(empty)}/{len(trajectories)} trajectories ({frac:.1%}) contain no "
        "extracted action -- these are failed collections, not hard instances, "
        "and must be excluded rather than analysed",
    )


def check_availability_leak(
    report: Report, trajectories: Sequence[dict[str, Any]]
) -> None:
    """Flags a feature family whose *presence* predicts the outcome.

    This is leakage of a kind provenance alone cannot see: each recorded value
    may be a genuine measurement, yet if the rows that carry the column are also
    the rows that succeed, a model handed that column separates the classes by
    detecting its own availability. The imputation used to fill the gap becomes
    the discriminative signal.

    We hit exactly this: token features survived on 25 of 225 trajectories, and
    all 11 successes lay inside those 25. Every family then scored AUROC above
    0.9 while measuring nothing but which rows had been collected successfully.
    """
    labelled = [t for t in trajectories if t["label"] is not None]
    if len(labelled) < 10:
        return
    total_pos = sum(t["label"] for t in labelled)
    if total_pos in (0, len(labelled)):
        return  # single-class overall; nothing to confound

    for family in ("logprobs", "confidence", "status_code"):
        have = [t for t in labelled if t["available"].get(family)]
        lack = [t for t in labelled if not t["available"].get(family)]
        if not have or not lack:
            continue
        p_have = sum(t["label"] for t in have) / len(have)
        p_lack = sum(t["label"] for t in lack) / len(lack)
        # Share of all positives that fall inside the rows carrying the column.
        concentration = sum(t["label"] for t in have) / total_pos
        if abs(p_have - p_lack) < 0.15:
            continue
        severity = "fatal" if concentration > 0.90 or concentration < 0.10 else "warn"
        report.add(
            "availability-leak", family, severity,
            f"presence of this column predicts the label: Pass@1 {p_have:.2f} on "
            f"the {len(have)} rows that have it against {p_lack:.2f} on the "
            f"{len(lack)} that do not, holding {concentration:.0%} of all "
            "positives -- a model given this family can separate the classes by "
            "detecting availability",
        )


def check_imputed(report: Report, column: str, flags: Sequence[Any]) -> None:
    if not flags:
        report.add(
            "imputed", column, "warn",
            "no provenance flag recorded -- cannot verify this column was measured",
        )
        return
    bad = sum(1 for f in flags if f in UNMEASURED)
    if bad == len(flags):
        report.add(
            "imputed", column, "fatal",
            f"never measured on any of {len(flags)} rows",
        )
    elif bad:
        report.add(
            "imputed", column, "warn",
            f"not measured on {bad}/{len(flags)} rows ({bad/len(flags):.1%}) -- "
            "exclude those rows rather than imputing",
        )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def validate(telemetry_dir: Path) -> Report:
    """Validates every JSONL trajectory under ``telemetry_dir``."""
    report = Report()

    logprobs: list[float] = []
    confidences: list[Any] = []
    observations: list[str] = []
    status_codes: list[Any] = []
    prov_logprobs: list[Any] = []
    prov_conf: list[Any] = []
    prov_status: list[Any] = []

    trajectories: list[dict[str, Any]] = []

    for path in sorted(Path(telemetry_dir).rglob("*.jsonl")):
        steps = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not steps:
            continue
        report.n_trajectories += 1
        report.n_steps += len(steps)

        # Trajectory-level record, for the two checks that cannot be made on a
        # flattened step stream.
        provs = [s.get("provenance", {}) or {} for s in steps]
        trajectories.append({
            "scenario": path.stem,
            "n_actions": sum(1 for s in steps if s.get("action")),
            "label": (steps[-1].get("task_success")
                      if "task_success" in steps[-1] else None),
            "available": {
                "logprobs": any(p.get("logprobs") == "provider" for p in provs),
                "confidence": any(p.get("confidence") == "parsed" for p in provs),
                "status_code": any(p.get("status_code") == "mcp_response"
                                   for p in provs),
            },
        })

        for step in steps:
            prov = step.get("provenance", {}) or {}
            logprobs.extend(step.get("thought_logprobs") or [])
            confidences.append(step.get("verbalized_confidence"))
            observations.append(str(step.get("observation", "")))
            prov_logprobs.append(prov.get("logprobs"))
            prov_conf.append(prov.get("confidence"))
            for call in step.get("mcp_tool_calls", []) or []:
                status_codes.append(call.get("status_code"))
                prov_status.append(call.get("source") or prov.get("status_code"))

    if report.n_trajectories == 0:
        report.add("imputed", "*", "fatal", f"no trajectories found under {telemetry_dir}")
        return report

    # Trajectory-level first: a corpus of failed collections should be rejected
    # before anyone reads its column statistics.
    check_degenerate(report, trajectories)
    check_availability_leak(report, trajectories)

    check_synthetic(report, "thought_logprobs", logprobs)
    check_constant(report, "verbalized_confidence", confidences)
    check_leakage(report, "observation", observations)
    check_constant(report, "mcp_status_code", status_codes, warn_low_cardinality=False)
    check_imputed(report, "thought_logprobs", prov_logprobs)
    check_imputed(report, "verbalized_confidence", prov_conf)
    check_imputed(report, "mcp_status_code", prov_status)

    return report


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("telemetry_dir", type=Path)
    args = ap.parse_args()

    report = validate(args.telemetry_dir)
    print(report.render())
    # Exit non-zero only when nothing measured survives. A column-level defect
    # is reported but does not block a study that excludes that column.
    return 1 if report.corpus_unusable else 0


if __name__ == "__main__":
    raise SystemExit(main())
