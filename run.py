#!/usr/bin/env python3
"""
AutoVoiceEvals — single command entry point.

Usage:
  python run.py                    # Run with default config.yaml
  python run.py --config my.yaml   # Run with custom config
"""

import argparse
import sys

from src.config import load_config
from src.runner import run


def main():
    parser = argparse.ArgumentParser(
        description="AutoVoiceEvals: Autoresearch for Voice AI Agent Testing",
    )
    parser.add_argument(
        "--config", "-c", default=None,
        help="Path to config YAML (default: config.yaml)",
    )
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)

    run(cfg)


if __name__ == "__main__":
    main()
