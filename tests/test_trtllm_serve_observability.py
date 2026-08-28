# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observability capture for trtllm_serve stacks: tachometer targets + perf_metrics dump."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from srtctl.cli.mixins.frontend_stage import FrontendTopology
from srtctl.core.schema import TachometerConfig
from srtctl.core.telemetry import generate_tachometer_config
from srtctl.core.topology import Process


def _procs():
    return [
        Process("node-a", frozenset(range(4)), 7500, 6100, "prefill", 0, node_rank=0),
        Process("node-b", frozenset(range(4)), 7501, 0, "prefill", 0, node_rank=1),  # follower, no http
        Process("node-c", frozenset(range(4)), 7502, 6100, "decode", 0, node_rank=0),
    ]


@patch("srtctl.core.telemetry.get_hostname_ip", return_value="10.0.0.1")
def test_trtllm_serve_targets_use_http_port_and_prometheus_path(_ip):
    """trtllm-serve's /metrics is JSON iteration stats; the Prometheus surface is
    /prometheus/metrics on the worker HTTP port (mounted when the engine runs
    with return_perf_metrics), and unconditionally on the disagg orchestrator."""
    runtime = MagicMock(job_id="1", run_name="t", network_interface="eth0")
    runtime.log_dir = Path("/runs/1/logs")
    topology = FrontendTopology(nginx_node=None, frontend_nodes=["node-h"], frontend_port=8000, public_port=8000)

    text = generate_tachometer_config(
        processes=_procs(),
        frontend_topology=topology,
        runtime=runtime,
        tachometer=TachometerConfig(enabled=True),
        frontend_type="trtllm_serve",
    )

    assert 'url = "http://10.0.0.1:6100/prometheus/metrics"' in text
    assert 'url = "http://10.0.0.1:8000/prometheus/metrics"' in text
    # sys-ports and bare /metrics paths must not appear for this stack
    assert ":7500" not in text and ":7501" not in text and ":7502" not in text
    assert '6100/metrics"' not in text and '8000/metrics"' not in text
    # the follower (http_port 0) is not a target
    assert text.count('filter = "backend"') == 2


@patch("srtctl.core.telemetry.get_hostname_ip", return_value="10.0.0.1")
def test_dynamo_targets_unchanged(_ip):
    runtime = MagicMock(job_id="1", run_name="t", network_interface="eth0")
    runtime.log_dir = Path("/runs/1/logs")
    topology = FrontendTopology(nginx_node=None, frontend_nodes=["node-h"], frontend_port=8000, public_port=8000)
    text = generate_tachometer_config(
        processes=_procs(),
        frontend_topology=topology,
        runtime=runtime,
        tachometer=TachometerConfig(enabled=True),
        frontend_type="dynamo",
    )
    assert 'url = "http://10.0.0.1:7500/metrics"' in text
    assert 'url = "http://10.0.0.1:8000/metrics"' in text
    assert "/prometheus/metrics" not in text


class TestPerfMetricsDump:
    @staticmethod
    def _stage(tmp_path):
        from srtctl.cli.mixins.benchmark_stage import BenchmarkStageMixin

        class Stage(BenchmarkStageMixin):
            @property
            def backend_processes(self):
                return _procs()

            def _public_api_node(self):
                return "node-h"

        stage = Stage()
        stage.runtime = MagicMock()
        stage.runtime.log_dir = tmp_path
        stage.runtime.frontend_port = 8000
        stage.runtime.network_interface = "eth0"
        return stage

    @patch("srtctl.cli.mixins.benchmark_stage.get_hostname_ip", side_effect=lambda n, i: f"ip-{n}")
    def test_dumps_frontend_and_worker_leaders(self, _ip, tmp_path):
        stage = self._stage(tmp_path)
        fetched = []

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def raise_for_status(self):
                pass

            def iter_content(self, chunk_size):
                return [b"{}"]

        with patch("requests.get", side_effect=lambda url, **kw: fetched.append(url) or _Resp()):
            stage._dump_trtllm_serve_perf_metrics()

        assert fetched == [
            "http://ip-node-h:8000/perf_metrics",
            "http://ip-node-a:6100/perf_metrics",
            "http://ip-node-c:6100/perf_metrics",
        ]
        assert (tmp_path / "perf_metrics_frontend.json").read_bytes() == b"{}"
        assert (tmp_path / "perf_metrics_prefill0.json").exists()
        assert (tmp_path / "perf_metrics_decode0.json").exists()

    @patch("srtctl.cli.mixins.benchmark_stage.get_hostname_ip", side_effect=lambda n, i: f"ip-{n}")
    def test_fetch_failure_is_best_effort(self, _ip, tmp_path):
        stage = self._stage(tmp_path)
        with patch("requests.get", side_effect=OSError("connection refused")):
            stage._dump_trtllm_serve_perf_metrics()  # must not raise
        assert not list(tmp_path.glob("perf_metrics_*.json"))


def test_dynamo_hash_build_honors_repo_url():
    """`dynamo.repo_url` steers the in-job source build to a fork; the default
    stays on ai-dynamo (previously required hand-patching the clone URL on the
    cluster checkout)."""
    from srtctl.core.schema import DynamoConfig

    fork = DynamoConfig(install=True, hash="8f2d22ff9244", repo_url="https://github.com/nv-yna/dynamo.git")
    cmd = fork.get_install_commands()
    assert "git clone https://github.com/nv-yna/dynamo.git dynamo" in cmd
    assert "git checkout 8f2d22ff9244" in cmd

    default = DynamoConfig(install=True, hash="8f2d22ff9244")
    assert "git clone https://github.com/ai-dynamo/dynamo.git dynamo" in default.get_install_commands()
