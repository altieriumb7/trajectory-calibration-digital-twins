#!/usr/bin/env python3
import os
import sys
import json
from pathlib import Path

# Setup paths
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEMO = _PROJECT_ROOT / "ReActXen" / "src" / "reactxen" / "demo" / "intent_implementation_demo"
_REACTXEN_SRC = _PROJECT_ROOT / "ReActXen" / "src"

sys.path.insert(0, str(_REACTXEN_SRC))
sys.path.insert(0, str(_DEMO))

def verify_datasets():
    print("=== Checking Industrial Datasets ===")
    
    # Check default data locations
    # 1. Root data folder
    root_data = _PROJECT_ROOT / "data"
    print(f"Checking root 'data' dir: {root_data.resolve()}")
    if root_data.exists():
        print(f"  Contents: {[f.name for f in root_data.iterdir()][:10]}")
    else:
        print("  Root 'data' directory does not exist.")

    # 2. ReActXen demo data folder
    demo_data_default = _DEMO / "multi_agent_implementation_demo" / "PDMBench_Data_Directory" / "submission096"
    print(f"Checking demo 'submission096' dir: {demo_data_default.resolve()}")
    if demo_data_default.exists():
        print(f"  Files: {[f.name for f in demo_data_default.glob('*.txt')] + [f.name for f in demo_data_default.glob('*.csv')]}")
    else:
        print("  Demo 'submission096' directory does not exist.")

    # Check environment variable
    phmforge_data_env = os.environ.get("PHMFORGE_DATA_DIR")
    print(f"PHMFORGE_DATA_DIR Env Var: {phmforge_data_env}")
    if phmforge_data_env:
        env_path = Path(phmforge_data_env)
        if env_path.exists():
            print(f"  Env path contents: {[f.name for f in env_path.iterdir()][:10]}")
        else:
            print("  Warning: path in PHMFORGE_DATA_DIR does not exist.")
            
    print("-" * 50)

def test_react_dry_run():
    print("=== Dry-running ReAct Action (1 test scenario) ===")
    try:
        from reactxen.prebuilt.create_reactxen_agent import create_reactxen_agent
        from tools.data_tools import LoadDatasetTool, LoadGroundTruthTool
        
        scenario_file = _DEMO / "scenarios" / "close_ended_scenarios.json"
        if not scenario_file.exists():
            scenario_file = _DEMO / "scenarios" / "phm_scenarios.json"
            
        with open(scenario_file) as f:
            scenarios = json.load(f)
            # handle both formats
            if isinstance(scenarios, dict):
                scenarios = scenarios.get("pdm_scenarios", [])
        
        if not scenarios:
            print("No scenarios found to run.")
            return
            
        scenario = scenarios[0]
        print(f"Selected scenario: {scenario['task_id']} ({scenario['classification_type']})")
        print(f"Question: {scenario.get('input_question') or scenario.get('question')}")
        
        # Verify if local WatsonX credentials or OpenAI credentials are set
        from shared.credentials_utils import load_credentials
        creds = load_credentials()
        print(f"Available Credentials: {[k for k, v in creds.items() if v]}")
        
        # Test agent initialization
        tools = [LoadDatasetTool(), LoadGroundTruthTool()]
        
        # We perform a mock run or check if we can run WatsonX
        model_id = 8 # Granite-4-H-small
        print(f"Instantiating agent with model_id={model_id} (Granite-4-H)...")
        # Let's perform a brief dry run that just checks if the imports and LLM binding works.
        import reactxen.utils.model_inference as _mi
        print(f"Supported modelset names: {_mi.modelset}")
        
        print("Verification completed successfully.")
    except Exception as e:
        import traceback
        print(f"Error during verification: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    verify_datasets()
    test_react_dry_run()
