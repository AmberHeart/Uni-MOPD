# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, MOPD contributors.
"""torchrun entry point for the independent five-teacher Megatron service."""
from swift.megatron.train.teacher_server import megatron_teacher_server_main


if __name__ == '__main__':
    megatron_teacher_server_main()
