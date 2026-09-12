"""GPU probe: check statuses, report schema, transports, evidence flow."""

import argparse
import json
import subprocess
import sys

import jsonschema
import pytest

from kernelagent import cli
from kernelagent.adapters.storage import EvidenceStore
from kernelagent.probe import (
    FAIL,
    PASS,
    UNAVAILABLE,
    _driver_api_sequence,
    _isolated_driver_check,
    _kernel_launch_sequence,
    build_report,
    check_ncu,
    check_nvcc,
    check_nvidia_smi,
)

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "schemas" / "gpu-probe.schema.json").read_text(encoding="utf-8"))


# -- command-backed checks (monkeypatched runner) ---------------------------


@pytest.fixture()
def fake_run(monkeypatch):
    def install(result):
        monkeypatch.setattr("kernelagent.probe._run_command", lambda argv: result)

    return install


def test_nvidia_smi_missing_is_unavailable_not_fail(fake_run):
    fake_run(None)
    check = check_nvidia_smi()
    assert check["status"] == UNAVAILABLE
    assert "not found" in check["detail"]["reason"]


def test_nvidia_smi_parses_gpu_identity(fake_run):
    fake_run((0, "RTX X, GPU-uuid-1, 610.62, 8.9, 8188 MiB\n", ""))
    check = check_nvidia_smi()
    assert check["status"] == PASS
    gpu = check["detail"]["gpus"][0]
    assert gpu["name"] == "RTX X"
    assert gpu["compute_capability"] == "8.9"
    assert gpu["driver_version"] == "610.62"


@pytest.mark.parametrize(
    ("result", "reason"),
    [
        ((1, "", "driver error"), "exit_code"),
        ((0, "only-one-field\n", ""), "unexpected csv shape"),
        ((0, "", ""), "no gpu rows"),
    ],
    ids=["tool-error", "malformed", "empty"],
)
def test_nvidia_smi_failures_are_explicit(fake_run, result, reason):
    fake_run(result)
    check = check_nvidia_smi()
    assert check["status"] == FAIL
    assert reason in json.dumps(check["detail"])


def test_nvcc_and_ncu_absent_is_unavailable(fake_run):
    fake_run(None)
    assert check_nvcc()["status"] == UNAVAILABLE
    assert check_ncu()["status"] == UNAVAILABLE


def test_ncu_version_parsed(fake_run):
    fake_run((0, "\nVersion 1.2.3\n", ""))
    check = check_ncu()
    assert check["status"] == PASS
    assert check["detail"]["version"] == "1.2.3"
    assert "T15" in check["detail"]["scope"]


# -- driver-backed checks (fake ctypes lib) ---------------------------------


class FakeCudaLib:
    def __init__(self, init=0, count=1, retain=0, load=0, launch=0, written=42):
        self.init, self.count, self.retain = init, count, retain
        self.load, self.launch, self.written = load, launch, written

    def cuInit(self, flags):
        return self.init

    def cuDeviceGetCount(self, ref):
        ref._obj.value = self.count
        return 0

    def cuDeviceGet(self, ref, index):
        return 0

    def cuDeviceGetName(self, buf, size, device):
        target = getattr(buf, "_obj", buf)
        target.value = b"FakeGPU"
        return 0

    def cuDeviceComputeCapability(self, major, minor, device):
        major._obj.value, minor._obj.value = 8, 9
        return 0

    def cuDevicePrimaryCtxRetain(self, ctx, device):
        ctx._obj.value = 1234
        return self.retain

    def cuCtxSetCurrent(self, ctx):
        return 0

    def cuModuleLoadData(self, module, image):
        module._obj.value = 555
        return self.load

    def cuModuleGetFunction(self, fn, module, name):
        return 0

    def cuMemAlloc(self, ptr, size):
        ptr._obj.value = 999
        return 0

    def cuLaunchKernel(self, fn, gx, gy, gz, bx, by, bz, shared, stream, params, extra):
        return self.launch

    def cuCtxSynchronize(self):
        return 0

    def cuMemcpyDtoH(self, dst, src, size):
        dst._obj.value = self.written
        return 0

    def cuMemFree(self, ptr):
        return 0

    def cuModuleUnload(self, module):
        return 0

    def cuDevicePrimaryCtxRelease(self, device):
        return 0


@pytest.fixture()
def fake_lib(monkeypatch):
    def install(lib):
        monkeypatch.setattr("kernelagent.probe._load_cuda_driver_lib", lambda: lib)

    return install


def test_driver_api_pass_reports_device(fake_lib):
    fake_lib(FakeCudaLib())
    check = _driver_api_sequence(FakeCudaLib())
    assert check["status"] == PASS
    assert check["detail"]["device_0"] == "FakeGPU"
    assert check["detail"]["compute_capability"] == "8.9"


def test_driver_api_zero_devices_is_unavailable(fake_lib):
    assert _driver_api_sequence(FakeCudaLib(count=0))["status"] == UNAVAILABLE


def test_driver_api_init_failure_is_fail(fake_lib):
    assert _driver_api_sequence(FakeCudaLib(init=0x200))["status"] == FAIL


def test_kernel_launch_pass_on_real_round_trip(fake_lib):
    fake_lib(FakeCudaLib())
    check = _kernel_launch_sequence(FakeCudaLib())
    assert check["status"] == PASS
    assert "42" in check["detail"]["proof"]


def test_kernel_launch_wrong_device_value_is_fail(fake_lib):
    fake_lib(FakeCudaLib(written=7))
    check = _kernel_launch_sequence(FakeCudaLib(written=7))
    assert check["status"] == FAIL
    assert "7" in check["detail"]["reason"]


def test_kernel_launch_ptx_jit_failure_is_fail(fake_lib):
    fake_lib(FakeCudaLib(load=0x219))
    check = _kernel_launch_sequence(FakeCudaLib(load=0x219))
    assert check["status"] == FAIL
    assert "0x219" in json.dumps(check["detail"])


def test_isolated_check_parses_child_json(monkeypatch):
    payload = "KERNELAGENT_PROBE_JSON_BEGIN\n" + '{"status": "pass", "detail": {}}'

    def fake_run(command, capture_output=True, text=True, timeout=None):
        return subprocess.CompletedProcess(command, 0, stdout=payload, stderr="")

    monkeypatch.setattr("kernelagent.probe.subprocess.run", fake_run)
    result = _isolated_driver_check("driver_api")
    assert result["status"] == PASS


def test_isolated_check_reports_child_crash(monkeypatch):
    def fake_run(command, capture_output=True, text=True, timeout=None):
        return subprocess.CompletedProcess(command, -11, stdout="", stderr="Segmentation fault")

    monkeypatch.setattr("kernelagent.probe.subprocess.run", fake_run)
    result = _isolated_driver_check("kernel_launch")
    assert result["status"] == FAIL
    assert "crashed" in result["detail"]["reason"]


# -- report, schema, exit semantics -----------------------------------------


def test_report_matches_schema_and_summary(monkeypatch):
    monkeypatch.setattr("kernelagent.probe._load_cuda_driver_lib", lambda: None)
    monkeypatch.setattr("kernelagent.probe._run_command", lambda argv: None)
    report = build_report()
    jsonschema.validate(report, SCHEMA)
    counted = {status: 0 for status in (PASS, FAIL, UNAVAILABLE)}
    for check in report["checks"]:
        counted[check["status"]] += 1
    assert report["summary"] == counted


def test_probe_module_imports_no_gpu_stack():
    code = (
        "import sys; import kernelagent.probe; "
        "bad = {'torch', 'triton', 'openai', 'anthropic', 'transformers'}; "
        "assert not bad & sys.modules.keys(), sorted(bad & sys.modules.keys())"
    )
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr


# -- CLI orchestration -------------------------------------------------------


def _probe_args(tmp_path, target="native", evidence_root=None):
    return argparse.Namespace(
        target=target,
        wsl_distro=None,
        output=tmp_path / "probe",
        evidence_root=evidence_root,
    )


def test_cli_native_round_trip(monkeypatch, tmp_path):
    report = {
        "schema_version": "1.0",
        "probe_protocol": "gpu-probe-v1",
        "generated_at": "2026-09-12T00:00:00+00:00",
        "platform": {"system": "Fake", "release": "1", "machine": "x86"},
        "python": "3.13.3",
        "checks": [{"name": "nvidia_smi", "status": "unavailable", "detail": {}}],
        "summary": {"pass": 0, "fail": 0, "unavailable": 1},
    }
    monkeypatch.setattr(cli, "_run_probe_native", lambda: (0, json.dumps(report), ""))
    code = cli._probe_command(_probe_args(tmp_path))
    assert code == 0
    saved = json.loads((tmp_path / "probe" / "gpu-probe-report.json").read_text(encoding="utf-8"))
    assert saved == report


def test_cli_records_report_into_evidence_store(monkeypatch, tmp_path):
    report = {
        "schema_version": "1.0",
        "probe_protocol": "gpu-probe-v1",
        "generated_at": "2026-09-12T00:00:00+00:00",
        "platform": {"system": "Fake", "release": "1", "machine": "x86"},
        "python": "3.13.3",
        "checks": [{"name": "cuda_driver_api", "status": "fail", "detail": {"reason": "x"}}],
        "summary": {"pass": 0, "fail": 1, "unavailable": 0},
    }
    monkeypatch.setattr(cli, "_run_probe_native", lambda: (1, json.dumps(report), ""))
    evidence_root = tmp_path / "evidence"
    code = cli._probe_command(_probe_args(tmp_path, evidence_root=evidence_root))
    assert code == 1  # failing check propagates
    with EvidenceStore(evidence_root) as store:
        assert store.audit().healthy
        assert store.audit().indexed == 1
        kinds = [event["kind"] for event in store.events()]
        assert kinds == []


def test_cli_wsl_transport_feeds_script_via_stdin(monkeypatch, tmp_path):
    captured = {}

    def fake_run(command, input=None, capture_output=True, text=True, timeout=None):
        captured["command"] = command
        captured["input"] = input
        return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    args = argparse.Namespace(
        target="wsl", wsl_distro="Ubuntu-24.04", output=tmp_path / "p", evidence_root=None
    )
    cli._probe_command(args)
    assert captured["command"][:2] == ["wsl.exe", "-d"]
    assert captured["command"][-3:-1] == ["bash", "-c"]
    assert "cat >" in captured["command"][-1] and "python3" in captured["command"][-1]
    assert "probe_fill" in captured["input"]
