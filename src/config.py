"""Load configuration from config.yaml and .env."""

import os
import yaml
from dotenv import load_dotenv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def load_config(path: str = None) -> dict:
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Inject secrets from env
    cfg["secrets"] = {
        "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
        "vapi_api_key": os.environ.get("VAPI_API_KEY", ""),
    }

    if not cfg["secrets"]["anthropic_api_key"]:
        raise ValueError("ANTHROPIC_API_KEY not set in .env or environment")
    if not cfg["secrets"]["vapi_api_key"]:
        raise ValueError("VAPI_API_KEY not set in .env or environment")

    cfg["output"]["dir"] = str(ROOT / cfg["output"]["dir"])
    return cfg
