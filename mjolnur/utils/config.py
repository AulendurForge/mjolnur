# mjolnur/utils/config.py
import yaml
from typing import Dict, Any


def load_config(config_path: str = None, overrides: Dict[str, Any] = None) -> dict:
    """Load config from file with optional overrides for experimentation"""
    if config_path:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    # Apply overrides (supports nested dicts)
    if overrides:

        def update_nested(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_nested(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        cfg = update_nested(cfg, overrides)

    return cfg
