#!/usr/bin/env python3
"""Command-line entry point for Constrained Branch Suppression training."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import fields


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def configure_environment() -> tuple[str, bool]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--cuda_visible_devices",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    )
    parser.add_argument("--disable_qwen35_fast_path", action="store_true")
    args, _ = parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    disabled = args.disable_qwen35_fast_path or os.environ.get(
        "DISABLE_QWEN35_FAST_PATH", "0"
    ).strip().lower() in {"1", "true", "yes", "y"}
    os.environ["DISABLE_QWEN35_FAST_PATH"] = "1" if disabled else "0"
    return args.cuda_visible_devices, disabled


CUDA_VISIBLE_DEVICES, DISABLE_QWEN35_FAST_PATH = configure_environment()

if DISABLE_QWEN35_FAST_PATH:
    import transformers.utils.import_utils as import_utils

    import_utils.is_flash_linear_attention_available = lambda: False
    import_utils.is_causal_conv1d_available = lambda: False


from constrained_branch_suppression import (
    ConstrainedBranchSuppressionTrainer,
    SuppressionTrainingConfig,
)
from paired_response_dataset import PairedResponseDataset


ALIASES = {
    "model_path": ("--model_path", "--model"),
    "data_path": ("--data_path", "--dataset"),
    "val_data_path": ("--val_data_path", "--validation_data"),
    "output_dir": ("--output_dir", "--output"),
}


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    for field in fields(SuppressionTrainingConfig):
        default = field.default
        option_strings = ALIASES.get(field.name, (f"--{field.name}",))
        kwargs = {"dest": field.name, "default": None}
        if isinstance(default, bool):
            kwargs["type"] = str_to_bool
        elif isinstance(default, int):
            kwargs["type"] = int
        elif isinstance(default, float):
            kwargs["type"] = float
        else:
            kwargs["type"] = str
        parser.add_argument(*option_strings, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(
        description="FTTrap Constrained Branch Suppression trainer (single process, no DDP)"
    )
    parser.add_argument(
        "--cuda_visible_devices",
        default=CUDA_VISIBLE_DEVICES,
        help="Comma-separated physical GPU ids exposed to this one process.",
    )
    parser.add_argument(
        "--disable_qwen35_fast_path",
        action="store_true",
        help="Disable Qwen3.5 FLA/causal-conv fast kernels for compatibility.",
    )
    parser.add_argument(
        "--config_file",
        default="",
        help="Optional JSON object with SuppressionTrainingConfig overrides.",
    )
    parser.add_argument(
        "--validate_data_only",
        action="store_true",
        help="Validate the paired-response schema without loading a model.",
    )
    parser.add_argument(
        "--print_config",
        action="store_true",
        help="Print the effective configuration before running.",
    )
    add_config_arguments(parser)
    return parser.parse_args()


def build_config(args) -> SuppressionTrainingConfig:
    config = SuppressionTrainingConfig()
    valid_names = {field.name for field in fields(config)}
    if args.config_file:
        with open(args.config_file, "r", encoding="utf-8") as handle:
            overrides = json.load(handle)
        if not isinstance(overrides, dict):
            raise ValueError("config_file must contain one JSON object.")
        unknown = set(overrides) - valid_names
        if unknown:
            raise ValueError(f"Unknown config_file keys: {sorted(unknown)}")
        for name, value in overrides.items():
            setattr(config, name, value)
    for name in valid_names:
        value = getattr(args, name)
        if value is not None:
            setattr(config, name, value)
    return config


def validate_data(config: SuppressionTrainingConfig) -> None:
    train_dataset = PairedResponseDataset(
        config.data_path,
        sample_limit=config.sample_limit,
        sample_kinds={"poison_full"},
        verify_images=config.verify_images,
    )
    payload = {"train": train_dataset.summary()}
    if config.val_data_path:
        val_dataset = PairedResponseDataset(
            config.val_data_path,
            sample_limit=config.val_sample_limit,
            sample_kinds={"poison_full"},
            verify_images=config.verify_images,
        )
        payload["validation"] = val_dataset.summary()
    print(json.dumps(payload, indent=2, ensure_ascii=True), flush=True)


def main() -> int:
    args = parse_args()
    config = build_config(args)
    if args.print_config:
        from dataclasses import asdict

        print(json.dumps(asdict(config), indent=2, ensure_ascii=True), flush=True)
    if args.validate_data_only:
        validate_data(config)
        return 0
    result = ConstrainedBranchSuppressionTrainer(config).train()
    return 3 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
