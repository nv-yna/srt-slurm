# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""backend.worker_command_prefix wraps every TRT-LLM worker command verbatim."""

from pathlib import Path
from unittest.mock import MagicMock

from srtctl.backends import TRTLLMProtocol, TRTLLMServerConfig
from srtctl.core.topology import Process


def _proc():
    return Process(
        node="node0",
        gpu_indices=frozenset({0, 1, 2, 3}),
        sys_port=7500,
        http_port=6100,
        endpoint_mode="decode",
        endpoint_index=0,
    )


def _runtime(tmp_path):
    rt = MagicMock()
    rt.worker_model_arg = "/model"
    rt.is_hf_model = False
    rt.model_path = Path("/lustre/DeepSeek-V4-Pro")
    rt.log_dir = Path(tmp_path)
    rt.gpu_type = "vr200"
    rt.request_plane = "tcp"
    return rt


def test_worker_command_prefix_leads_the_command(tmp_path):
    backend = TRTLLMProtocol(
        trtllm_config=TRTLLMServerConfig(decode={"tensor_parallel_size": 4}),
        worker_command_prefix=("bash", "/configs/pyspy_wrap.sh"),
    )
    cmd = backend.build_worker_command(_proc(), [_proc()], _runtime(tmp_path))
    assert cmd[:2] == ["bash", "/configs/pyspy_wrap.sh"]
    # The wrapped command is intact behind the prefix.
    assert "trtllm-llmapi-launch" in cmd
    assert "dynamo.trtllm" in cmd


def test_default_is_no_prefix(tmp_path):
    backend = TRTLLMProtocol(trtllm_config=TRTLLMServerConfig(decode={"tensor_parallel_size": 4}))
    cmd = backend.build_worker_command(_proc(), [_proc()], _runtime(tmp_path))
    assert cmd[0] == "trtllm-llmapi-launch"


def test_wrapper_script_ships():
    script = Path(__file__).resolve().parents[1] / "configs" / "pyspy_wrap.sh"
    assert script.exists()
    text = script.read_text()
    # The wrapper must never be the reason a worker fails to start.
    assert 'exec "$@"' in text
