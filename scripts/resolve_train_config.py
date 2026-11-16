# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0
"""Resolve a full training config from an optional public base config."""
import argparse
from pathlib import Path

from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()

    overlay = OmegaConf.load(args.config)
    base_path = overlay.pop('defaults', None)
    if base_path is None:
        resolved = overlay
    else:
        base_path = Path(base_path)
        if not base_path.is_absolute():
            base_path = args.config.resolve().parent / base_path
        resolved = OmegaConf.merge(OmegaConf.load(base_path), overlay)
    OmegaConf.save(resolved, args.output)


if __name__ == '__main__':
    main()
