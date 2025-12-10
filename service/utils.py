from typing import Dict, Any

import yaml


def load_yml_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)