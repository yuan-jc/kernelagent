"""T04 GPU acceptance smoke: one known CUDA request through the real
container boundary (ADR-0001), verified by the trusted parent.

Two frozen cases run against the pinned digest-pinned image with the GPU
attached by CDI device name:

- positive: a PTX JIT kernel writes 42 into device memory; the payload
  copies it back and records it. The parent independently requires the
  value to equal its own frozen constant and re-checks that the trusted
  input bytes never changed.
- negative: the same kernel writes 43. The payload exits 0 (the process
  itself "succeeded"), but the parent's verification must reject it - a
  candidate's exit code or artifact can never certify correctness.

The combined report is written to <output>/container-gpu-smoke.json. Exit
code 0 requires the positive case verified AND the negative case rejected."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from kernelagent.worker import ContainerSpec, WorkerRequest, execute_container

PYTHON_IMAGE = "docker.m.daocloud.io/library/python"
PYTHON_DIGEST = "sha256:528257d48c1da0dcecc2e725d1ae34498d60c965f1241e39cd6a85a8859bdf84"
PROTOCOL = "t04-container-gpu-smoke-v1"
EXPECTED_VALUE = 42

# The payload is parent-authored frozen code: a CUDA Driver API smoke that
# JIT-compiles the PTX, launches it on device 0, copies the result back and
# records it. _WRITE_VALUE_TOKEN_ is substituted per case; the parent keeps
# its own copy of the expected value and never trusts the payload alone.
_PAYLOAD_TEMPLATE = """# Container GPU smoke payload (frozen by the parent); stdlib only.
import ctypes
import json

_WRITE_VALUE = _WRITE_VALUE_TOKEN_

_PTX = (
    "\\n.version 7.0"
    "\\n.target sm_70"
    "\\n.address_size 64"
    "\\n.visible .entry smoke_fill(.param .u64 out_ptr)"
    "\\n{"
    "\\n    .reg .u64 %rd1;"
    "\\n    .reg .u32 %r1;"
    "\\n    ld.param.u64 %rd1, [out_ptr];"
    "\\n    mov.u32 %r1, " + str(_WRITE_VALUE) + ";"
    "\\n    st.global.u32 [%rd1], %r1;"
    "\\n    ret;"
    "\\n}"
)

trace = {}

def _step(name, curesult):
    trace[name] = "0x%x" % (curesult & 0xffffffff)
    return curesult == 0

def _record(value, error=""):
    json.dump({"device_value": value, "trace": trace, "error": error},
              open("/out/result.json", "w"))

lib = ctypes.CDLL("libcuda.so.1")
version = ctypes.c_int()
lib.cuDriverGetVersion(ctypes.byref(version))

class _CUdeviceptr(ctypes.c_uint64):
    pass

lib.cuInit.argtypes = [ctypes.c_uint]
lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
lib.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
lib.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
lib.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
lib.cuModuleGetFunction.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p]
lib.cuMemAlloc_v2.argtypes = [ctypes.POINTER(_CUdeviceptr), ctypes.c_size_t]
lib.cuMemFree_v2.argtypes = [_CUdeviceptr]
lib.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, _CUdeviceptr, ctypes.c_size_t]

try:
    if not _step("cuInit", lib.cuInit(0)):
        _record(None, "cuInit failed"); raise SystemExit(1)
    device = ctypes.c_int()
    if not _step("cuDeviceGet", lib.cuDeviceGet(ctypes.byref(device), 0)):
        _record(None, "cuDeviceGet failed"); raise SystemExit(1)
    context = ctypes.c_void_p()
    if not _step("cuDevicePrimaryCtxRetain",
                 lib.cuDevicePrimaryCtxRetain(ctypes.byref(context), device)):
        _record(None, "cuDevicePrimaryCtxRetain failed"); raise SystemExit(1)
    if not _step("cuCtxSetCurrent", lib.cuCtxSetCurrent(context)):
        _record(None, "cuCtxSetCurrent failed"); raise SystemExit(1)
    module = ctypes.c_void_p()
    if not _step("cuModuleLoadData", lib.cuModuleLoadData(ctypes.byref(module), _PTX.encode())):
        _record(None, "cuModuleLoadData (PTX JIT) failed"); raise SystemExit(1)
    function = ctypes.c_void_p()
    if not _step("cuModuleGetFunction",
                 lib.cuModuleGetFunction(ctypes.byref(function), module, b"smoke_fill")):
        _record(None, "cuModuleGetFunction failed"); raise SystemExit(1)
    buffer = _CUdeviceptr()
    if not _step("cuMemAlloc_v2", lib.cuMemAlloc_v2(ctypes.byref(buffer), 4)):
        _record(None, "cuMemAlloc_v2 failed"); raise SystemExit(1)
    try:
        kernel_arg = _CUdeviceptr(buffer.value)
        params = (ctypes.c_void_p * 1)(ctypes.cast(ctypes.byref(kernel_arg), ctypes.c_void_p))
        launch = lib.cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, None, params, None)
        if not _step("cuLaunchKernel", launch):
            _record(None, "cuLaunchKernel returned CUresult 0x%x" % (launch & 0xffffffff))
            raise SystemExit(1)
        sync = lib.cuCtxSynchronize()
        if not _step("cuCtxSynchronize", sync):
            _record(None, "cuCtxSynchronize returned CUresult 0x%x" % (sync & 0xffffffff))
            raise SystemExit(1)
        host_value = ctypes.c_uint32(0)
        if not _step("cuMemcpyDtoH_v2",
                     lib.cuMemcpyDtoH_v2(ctypes.byref(host_value), buffer, 4)):
            _record(None, "cuMemcpyDtoH_v2 failed"); raise SystemExit(1)
    finally:
        lib.cuMemFree_v2(buffer)
    _record(host_value.value)
    print("driver_version", version.value)
    raise SystemExit(0)
except SystemExit:
    raise
except Exception as exc:  # noqa: BLE001 - payload reports its own failure mode
    _record(None, "%s: %s" % (type(exc).__name__, exc))
    raise SystemExit(1)
"""


def _host_gpu_identity() -> dict:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except OSError:
        return {"nvidia_smi": "unavailable"}
    if completed.returncode != 0:
        return {"nvidia_smi": f"failed rc={completed.returncode}"}
    name, uuid, driver = [part.strip() for part in completed.stdout.strip().split(",")]
    return {"gpu_name": name, "gpu_uuid": uuid, "driver": driver}


def _dir_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _run_case(name: str, write_value: int, workspace: Path, gpu_device: str) -> dict:
    inputs = workspace / f"input-{name}"
    inputs.mkdir(parents=True, exist_ok=True)
    (inputs / "expected.json").write_text(
        json.dumps({"case": name}), encoding="utf-8"
    )  # content is irrelevant to the kernel; proves read-only mounting
    payload_path = inputs / "gpu_smoke_payload.py"
    payload_path.write_text(
        _PAYLOAD_TEMPLATE.replace("_WRITE_VALUE_TOKEN_", str(write_value)), encoding="utf-8"
    )
    inputs_before = _dir_hashes(inputs)
    request = WorkerRequest(
        request_id=f"gpu-smoke-{name}",
        argv=("python3", "/task/gpu_smoke_payload.py"),
        timeout_seconds=120.0,
        workspace_root=workspace,
    )
    spec = ContainerSpec(
        image=PYTHON_IMAGE,
        image_digest=PYTHON_DIGEST,
        read_only_mounts=(("/task", inputs),),
        gpu_devices=(gpu_device,),
    )
    outcome = execute_container(request, spec)
    inputs_after = _dir_hashes(inputs)
    result: dict = {
        "outcome_status": outcome.status,
        "exit_code": outcome.exit_code,
        "killed": outcome.killed,
        "stderr_tail": outcome.stderr_tail[-2000:],
        "inputs_unchanged": inputs_before == inputs_after,
        "expected_value": EXPECTED_VALUE,
    }
    record_path = Path(outcome.workdir or "") / "container-record.json"
    if record_path.exists():
        result["container_record"] = json.loads(record_path.read_text(encoding="utf-8"))
    out_dir = Path(outcome.workdir or "") / "out"
    result_file = out_dir / "result.json"
    if result_file.exists():
        result["payload_result"] = json.loads(result_file.read_text(encoding="utf-8"))
        result["device_value"] = result["payload_result"].get("device_value")
    # Trusted-side verification: the parent's own frozen constant decides.
    result["verified"] = (
        outcome.status == "completed"
        and result.get("device_value") == EXPECTED_VALUE
        and result["inputs_unchanged"]
    )
    return result


def _default_gpu_device() -> str:
    """ADR-0001 allocates the GPU by UUID, not by mutable index; fall back
    to the CDI index name only when nvidia-smi cannot answer."""
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except OSError:
        return "nvidia.com/gpu=0"
    uuid = completed.stdout.strip().splitlines()[0].strip() if completed.stdout.strip() else ""
    return f"nvidia.com/gpu={uuid}" if uuid else "nvidia.com/gpu=0"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/t04-gpu"), help="Report output directory"
    )
    parser.add_argument(
        "--gpu-device",
        default=None,
        help="CDI device name (default: nvidia.com/gpu=<UUID> detected via nvidia-smi)",
    )
    args = parser.parse_args()
    if args.gpu_device is None:
        args.gpu_device = _default_gpu_device()

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()

    workspace = args.output / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    positive = _run_case("positive", EXPECTED_VALUE, workspace, args.gpu_device)
    negative = _run_case("negative", EXPECTED_VALUE + 1, workspace, args.gpu_device)

    report = {
        "protocol": PROTOCOL,
        "commit": commit,
        "image": f"{PYTHON_IMAGE}@{PYTHON_DIGEST}",
        "gpu_device": args.gpu_device,
        "host_gpu": _host_gpu_identity(),
        "cases": {"positive": positive, "negative": negative},
        "accepted": positive["verified"] and not negative["verified"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "container-gpu-smoke.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8", newline="\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report={report_path} accepted={report['accepted']}")
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    sys.exit(main())
