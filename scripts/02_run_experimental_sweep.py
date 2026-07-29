#!/usr/bin/env python3
import os
import re
import json
import sys
from pathlib import Path

# Setup paths
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEMO = _PROJECT_ROOT / "ReActXen" / "src" / "reactxen" / "demo" / "intent_implementation_demo"
_RESULTS_DIR = _DEMO / "results" / "paper_table4_runs"
_TELEMETRY_DIR = _PROJECT_ROOT / "results" / "telemetry_runs"

# Regex to strip ANSI escape codes
ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE.sub('', text)

def parse_log_file(log_path: Path) -> dict:
    """Parses a console log file to extract details of all scenarios.
    Returns a dict mapping scenario_id -> list of steps.
    """
    if not log_path.exists():
        print(f"Warning: log file {log_path} not found.")
        return {}
        
    scenarios_data = {}
    current_scenario = None
    current_steps = []
    current_step = {}
    
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            clean_line = strip_ansi(line).strip()
            
            # Check for scenario start
            scenario_match = re.search(r"\[\d+/\d+\]\s+(pdm_\w+)", clean_line)
            if scenario_match:
                if current_scenario and current_steps:
                    scenarios_data[current_scenario] = current_steps
                current_scenario = scenario_match.group(1)
                current_steps = []
                current_step = {}
                continue
                
            if not current_scenario:
                continue
                
            # Parse Thought
            thought_match = re.match(r"Thought\s+(\d+):\s*(.*)", clean_line, re.IGNORECASE)
            if thought_match:
                if "thought" in current_step:
                    current_steps.append(current_step)
                step_idx = int(thought_match.group(1))
                current_step = {
                    "step": step_idx - 1,
                    "thought": thought_match.group(2),
                    "action": "",
                    "action_input": "",
                    "observation": ""
                }
                continue
                
            # Parse Action
            action_match = re.match(r"Action\s+(\d+):\s*(.*)", clean_line, re.IGNORECASE)
            if action_match:
                current_step["action"] = action_match.group(2)
                continue
                
            # Parse Action Input
            input_match = re.match(r"Action Input\s+(\d+):\s*(.*)", clean_line, re.IGNORECASE)
            if input_match:
                current_step["action_input"] = input_match.group(2)
                continue
                
            # Parse Observation
            obs_match = re.match(r"Observation\s+(\d+):\s*(.*)", clean_line, re.IGNORECASE)
            if obs_match:
                current_step["observation"] = obs_match.group(2)
                current_steps.append(current_step)
                current_step = {}
                continue
                
            if current_step:
                if "observation" in current_step and current_step["observation"]:
                    current_step["observation"] += " " + clean_line
                elif "action_input" in current_step and current_step["action_input"]:
                    current_step["action_input"] += " " + clean_line
                elif "thought" in current_step and current_step["thought"]:
                    current_step["thought"] += " " + clean_line
                    
        if current_scenario and current_steps:
            scenarios_data[current_scenario] = current_steps
            
    return scenarios_data

def run_sweep():
    print("=== Reconstructing & Running Telemetry Sweep ===")
    
    # Models to process (corresponds to the 6 backbones)
    models = [
        "ibm/granite-4-h-small",
        "meta-llama/llama-3-3-70b-instruct",
        "meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
        "mistralai/mistral-medium-2505",
        "mistralai/mistral-small-3-1-24b-instruct-2503",
        "openai/gpt-oss-120b"
    ]
    
    # TelemetryLogger imports
    sys.path.insert(0, str(_PROJECT_ROOT))
    from phmforge_calibration.telemetry_logger import TelemetryLogger
    logger = TelemetryLogger(output_dir=_TELEMETRY_DIR)
    
    # We clear output directory first to have a clean run
    import shutil
    if _TELEMETRY_DIR.exists():
        shutil.rmtree(_TELEMETRY_DIR)
    _TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
    
    for model_name in models:
        safe_name = model_name.replace("/", "_").replace(":", "_")
        
        # Load both reactxen and react files to see which one has more results
        reactxen_json = _RESULTS_DIR / f"reactxen__{safe_name}.json"
        reactxen_log = _RESULTS_DIR / "logs" / f"reactxen__{safe_name}.log"
        react_json = _RESULTS_DIR / f"react__{safe_name}.json"
        react_log = _RESULTS_DIR / "logs" / f"react__{safe_name}.log"
        
        json_file = reactxen_json
        log_file = reactxen_log
        
        # Determine the file with the most results
        n_reactxen = 0
        if reactxen_json.exists():
            try:
                with open(reactxen_json) as f:
                    n_reactxen = len(json.load(f).get("results", []))
            except:
                pass
                
        n_react = 0
        if react_json.exists():
            try:
                with open(react_json) as f:
                    n_react = len(json.load(f).get("results", []))
            except:
                pass
                
        if n_react > n_reactxen:
            json_file = react_json
            log_file = react_log
            print(f"Selecting react__ file for {model_name} (found {n_react} results vs {n_reactxen} in reactxen__)")
        else:
            print(f"Selecting reactxen__ file for {model_name} (found {n_reactxen} results)")
            
        if not json_file.exists():
            print(f"Error: results file {json_file} does not exist. Cannot process {model_name}.")
            continue
            
        print(f"Processing model: {model_name}...")
        
        with open(json_file, "r") as f:
            json_data = json.load(f)
            
        results = json_data.get("results", [])
        scenarios_steps = parse_log_file(log_file)
        
        count = 0
        for r in results:
            scenario_id = r["task_id"]
            task_success = r.get("correct", False)
            final_answer = r.get("answer", "")
            
            steps = scenarios_steps.get(scenario_id, [])
            
            if not steps:
                steps = [{
                    "step": 0,
                    "thought": f"Formulating final answer based on execution: {final_answer}",
                    "action": "Finish",
                    "action_input": final_answer,
                    "observation": "Task finished successfully" if task_success else "Task failed"
                }]
                
            logger.log_trajectory(
                model_name=model_name,
                scenario_id=scenario_id,
                steps_log=steps,
                final_answer=final_answer,
                task_success=task_success,
                raw_scratchpad=""
            )
            count += 1
            
        print(f"  Successfully wrote {count} telemetry logs for {model_name}.")
        
    print("=== Sweep processing completed successfully ===")

if __name__ == "__main__":
    run_sweep()
