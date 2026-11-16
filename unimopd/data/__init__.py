# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0
"""Query-pool preparation and allocation."""

from .allocate import UNIMOPD_DOMAINS, allocate_records, compute_quotas

__all__ = ['UNIMOPD_DOMAINS', 'allocate_records', 'compute_quotas']
