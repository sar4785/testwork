import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_config(path="configs/config.yaml"):
    with open(PROJECT_ROOT / path, "r") as f:
        return yaml.safe_load(f)

config = load_config()