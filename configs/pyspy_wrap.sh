#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Diagnostic wrapper (see backends/trtllm.py worker_command_prefix): sample the
# worker's rank-0 process with py-spy for the whole run. Rank 0 is where the
# TRT-LLM executor loop lives; hecate's yama ptrace_scope=1 forbids attaching
# to an already-running process, so py-spy must be the parent.
#
# --idle is load-bearing: the phenomenon under investigation is a BLOCKED
# executor thread, which only shows up when idle threads are sampled.
# --subprocesses follows the trtllm-llmapi-launch -> python exec/fork chain.
# Rate stays low (default 15 Hz) to bound sampling pauses on the many-thread
# rank-0 process.
#
# Non-rank-0 tasks, or a missing py-spy binary, exec straight through: this
# wrapper must never be the reason a worker fails to start.
if [ "${SLURM_PROCID:-1}" != "0" ] || [ ! -x /configs/py-spy ]; then
    exec "$@"
fi

out="/logs/pyspy_$(hostname -s).speedscope"
echo "[pyspy_wrap] sampling rank 0 on $(hostname -s) at ${PYSPY_RATE:-15} Hz -> ${out}" >&2
exec /configs/py-spy record \
    --format speedscope \
    --rate "${PYSPY_RATE:-15}" \
    --idle \
    --subprocesses \
    -o "${out}" \
    -- "$@"
