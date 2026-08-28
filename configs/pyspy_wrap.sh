#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -uo pipefail

# Diagnostic wrapper (see backends/trtllm.py worker_command_prefix): sample the
# worker's long-lived rank-0 python with py-spy.
#
# Why attach-by-pid instead of `py-spy record -- <cmd>`: trtllm-llmapi-launch
# runs the task in a BACKGROUND SUBSHELL with a scrubbed environment, so the
# wrapper's direct child is never the engine python and parent-mode py-spy
# stops after one sample ("process exited", observed on hecate job 489551).
# yama ptrace_scope=1 forbids attaching to non-descendants, but this wrapper
# IS an ancestor of the engine python, so --pid attach from here is allowed.
#
# Non-rank-0 tasks, or a missing py-spy binary, exec straight through: this
# wrapper must never be the reason a worker fails to start.
if [ "${SLURM_PROCID:-1}" != "0" ] || [ ! -x /configs/py-spy ]; then
    exec "$@"
fi

"$@" &
child=$!
trap 'kill -TERM "$child" 2>/dev/null' TERM INT

descendants() {  # all descendant pids of $1, breadth-first
    local frontier=("$1") out=() pid kids
    while [ "${#frontier[@]}" -gt 0 ]; do
        pid="${frontier[0]}"; frontier=("${frontier[@]:1}")
        kids=$(cat /proc/"$pid"/task/*/children 2>/dev/null || true)
        for k in $kids; do out+=("$k"); frontier+=("$k"); done
    done
    echo "${out[@]:-}"
}

(
    # Find THIS task's engine python (cmdline contains dynamo.trtllm or
    # trtllm_serve) among the child's descendants; the model load gives us a
    # wide window. Then attach for the remainder of the run.
    target=""
    for _ in $(seq 1 120); do
        sleep 5
        for pid in $(descendants "$child"); do
            if tr '\0' ' ' < /proc/"$pid"/cmdline 2>/dev/null | grep -qE "dynamo\.trtllm|trtllm[-_]serve"; then
                target="$pid"
                break 2
            fi
        done
    done
    if [ -z "$target" ]; then
        echo "[pyspy_wrap] no engine python found under pid $child after 600s; skipping" >&2
        exit 0
    fi
    out="/logs/pyspy_$(hostname -s).speedscope"
    echo "[pyspy_wrap] attaching py-spy to pid ${target} at ${PYSPY_RATE:-11} Hz -> ${out}" >&2
    /configs/py-spy record \
        --pid "$target" \
        --format speedscope \
        --rate "${PYSPY_RATE:-11}" \
        --idle \
        --nonblocking \
        -o "$out" >&2 || echo "[pyspy_wrap] py-spy exited nonzero (attach denied or target gone)" >&2
) &

wait "$child"
exit $?
