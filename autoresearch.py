"""
AutoVoiceEvals — Autoresearch Mode

Karpathy's autoresearch pattern applied to voice AI prompt optimization.
One artifact (system prompt), one metric (composite score), keep/revert,
run forever.

Usage:
    python autoresearch.py [--config config.yaml]
    python autoresearch.py --resume              # resume from last run

See program.md for the full protocol.
"""

import argparse
from src.config import load_config
from src import researcher


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AutoVoiceEvals — autoresearch mode")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last autoresearch.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    researcher.run(cfg, resume=args.resume)
