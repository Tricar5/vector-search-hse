from typing import Any

import yaml


def load_yml_config(config_path: str) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)
