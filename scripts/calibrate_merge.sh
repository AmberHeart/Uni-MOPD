#!/usr/bin/env bash
# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

python -m unimopd.merge.calibrate "$@"
