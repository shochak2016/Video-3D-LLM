"""Run a scanqa/sqa3d eval from a YAML config.

Usage:
    python scripts/3d/eval/run_from_config.py scripts/3d/configs/scanqa_default.yaml
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = REPO_ROOT / "scripts" / "3d" / "configs"
CKPT_DIR = REPO_ROOT / "ckpt"

TASK_TO_SCRIPT = {
    "scanqa": REPO_ROOT / "scripts" / "3d" / "eval" / "eval_scanqa.sh",
    "sqa3d": REPO_ROOT / "scripts" / "3d" / "eval" / "eval_sqa3d.sh",
}

VALID_PE = {"none", "mlp", "sinusoidal"}
VALID_COORD = {"center", "minmax", "average"}
VALID_SAMPLING = {"uniform", "mc-ratio90", "mc-ratio95"}


def resolve_checkpoint(pe: str, coord: str) -> dict:
    with open(CONFIGS_DIR / "checkpoints.yaml") as f:
        registry = yaml.safe_load(f)["checkpoints"]
    matches = [e for e in registry if e["pe"] == pe and e["coord"] == coord]
    if not matches:
        sys.exit(
            f"No checkpoint registered for pe={pe!r}, coord={coord!r}. "
            f"Add one to {CONFIGS_DIR / 'checkpoints.yaml'}."
        )
    if len(matches) > 1:
        sys.exit(f"Multiple checkpoints registered for pe={pe!r}, coord={coord!r}; pick one.")
    return matches[0]


def validate_checkpoint(entry: dict) -> None:
    cfg_path = CKPT_DIR / entry["name"] / "config.json"
    if not cfg_path.exists():
        sys.exit(f"Checkpoint {entry['name']!r} not found at {cfg_path}.")
    with open(cfg_path) as f:
        ckpt_cfg = json.load(f)
    actual = ckpt_cfg.get("world_position_embedding_type")
    expected = entry["world_position_embedding_type"]
    if actual != expected:
        sys.exit(
            f"Checkpoint {entry['name']!r} has world_position_embedding_type="
            f"{actual!r}, registry expected {expected!r}. Fix one of them."
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config", help="Path to YAML config")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    required = {"task", "frame_sampling_rate", "sampling_strategy", "pe", "coord"}
    missing = required - cfg.keys()
    if missing:
        sys.exit(f"Config missing keys: {sorted(missing)}")

    if cfg["task"] not in TASK_TO_SCRIPT:
        sys.exit(f"task must be one of {sorted(TASK_TO_SCRIPT)}, got {cfg['task']!r}")
    if cfg["pe"] not in VALID_PE:
        sys.exit(f"pe must be one of {sorted(VALID_PE)}, got {cfg['pe']!r}")
    if cfg["coord"] not in VALID_COORD:
        sys.exit(f"coord must be one of {sorted(VALID_COORD)}, got {cfg['coord']!r}")
    if cfg["sampling_strategy"] not in VALID_SAMPLING:
        sys.exit(
            f"sampling_strategy must be one of {sorted(VALID_SAMPLING)}, "
            f"got {cfg['sampling_strategy']!r}"
        )

    entry = resolve_checkpoint(cfg["pe"], cfg["coord"])
    validate_checkpoint(entry)

    script = TASK_TO_SCRIPT[cfg["task"]]
    cmd = [
        "bash",
        str(script),
        entry["name"],
        str(cfg["sampling_strategy"]),
        str(cfg["frame_sampling_rate"]),
    ]
    if cfg.get("test_size") is not None:
        cmd.append(str(cfg["test_size"]))

    print(f"[run_from_config] checkpoint: {entry['name']} ({entry['world_position_embedding_type']})")
    print(f"[run_from_config] cmd: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    main()
