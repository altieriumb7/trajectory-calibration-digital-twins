# --- V2 PROVENANCE GATE ----------------------------------------------------
# This script produced figures and tables for the original submission. Those
# artifacts are invalid: the token-level features were synthesised by the old
# telemetry logger, and the grading harness leaked its verdict into the
# observation text. Background: DigitalTwinsV2/AUDIT.md
#
# The gate below refuses to run on a corpus that cannot support the analysis.
# It is not a formality: on the current telemetry it exits non-zero.
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from phmforge_calibration.provenance import validate as _validate

_report = _validate(_Path(__file__).resolve().parent.parent / "results" / "telemetry_runs")
if not _report.ok:
    print(_report.render())
    print("")
    print("ABORTED: this corpus cannot support a calibration study.")
    print("Recapture first:  python DigitalTwinsV2/scripts/rerun_sweep.py capture ...")
    print("Background:       DigitalTwinsV2/AUDIT.md")
    raise SystemExit(1)
# --- END V2 PROVENANCE GATE ------------------------------------------------
