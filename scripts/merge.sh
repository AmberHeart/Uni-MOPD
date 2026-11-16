#!/usr/bin/env bash
# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 CONFIG.yaml" >&2
  exit 2
fi

python -m unimopd.merge.streaming --config "$1"
