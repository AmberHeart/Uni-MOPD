#!/usr/bin/env bash
# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
megatron_dir="${MEGATRON_DIR:-${repo_root}/Megatron-LM}"
megatron_commit="0d7e02bd26d3eb347ad8d1bef625002babfa6df7"
megatron_repo="https://github.com/NVIDIA/Megatron-LM.git"
patch_file="${repo_root}/patches/megatron-lm.patch"

if [[ ! -d "${megatron_dir}/.git" ]]; then
  git clone --filter=blob:none "${megatron_repo}" "${megatron_dir}"
fi

git -C "${megatron_dir}" fetch origin "${megatron_commit}"
current_commit="$(git -C "${megatron_dir}" rev-parse HEAD)"
if [[ "${current_commit}" != "${megatron_commit}" ]]; then
  git -C "${megatron_dir}" checkout --detach "${megatron_commit}"
fi

if git -C "${megatron_dir}" apply --reverse --check "${patch_file}" >/dev/null 2>&1; then
  echo "Uni-MOPD Megatron patch is already applied."
else
  git -C "${megatron_dir}" apply --check "${patch_file}"
  git -C "${megatron_dir}" apply "${patch_file}"
  echo "Applied Uni-MOPD Megatron patch."
fi

echo "Megatron-LM is ready at ${megatron_dir}."
echo "Set MEGATRON_LM_PATH=${megatron_dir} before training."
