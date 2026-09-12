"""Self-contained GPU environment probe (design 12, T03).

Runs inside the target environment and prints a versioned JSON report on
stdout. No kernelagent imports, no third-party dependencies: on a worker it
can be executed as `python3 - < probe.py` (e.g. fed through `wsl -e python3 -`).

Checks are independent and each reports an explicit status:
  pass         check succeeded
  fail         check ran and observed an error (malformed output, wrong result)
  unavailable  the tool/library this check needs is absent in this environment
Missing tools are never failures, and neither state is ever reported as pass.

The cuda_kernel_launch check is the small real probe: it loads the driver
library via ctypes, JIT-compiles an embedded PTX kernel, launches it on the
device, and verifies the written value over a device-memory round trip.
"""

from __future__ import annotations

import ctypes
import json
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROBE_PROTOCOL = "gpu-probe-v1"
SCHEMA_VERSION = "1.0"
_PROBE_MARKER = "KERNELAGENT_PROBE_JSON_BEGIN\n"
PASS, FAIL, UNAVAILABLE = "pass", "fail", "unavailable"

# Minimal PTX kernel: writes 42 to the first u32 of the output buffer.
# sm_70 PTX is forward-JIT-compiled by the driver for newer GPUs.
_PTX = """
.version 7.0
.target sm_70
.address_size 64
.visible .entry probe_fill(.param .u64 out_ptr)
{
    .reg .u64 %rd1;
    .reg .u32 %r1;
    ld.param.u64 %rd1, [out_ptr];
    mov.u32 %r1, 42;
    st.global.u32 [%rd1], %r1;
    ret;
}
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check(name: str, status: str, detail: dict | None = None, command: str | None = None) -> dict:
    if status not in (PASS, FAIL, UNAVAILABLE):
        raise ValueError(f"invalid check status {status!r}")
    entry = {"name": name, "status": status, "detail": detail or {}}
    if command:
        entry["command"] = command
    return entry


_EXTRA_TOOL_DIRS = ("/usr/lib/wsl/lib",)


def _run_command(argv: list[str]) -> tuple[int, str, str] | None:
    """Run a tool; None marks 'tool not present' (unavailable), never fail.
    Uses the resolved full path: bare names are not reliably re-resolved by
    CreateProcess (e.g. nvidia-smi.EXE), and WSL sessions may carry the tools
    in /usr/lib/wsl/lib without a login shell."""
    resolved = shutil.which(argv[0])
    if resolved is None:
        for extra in _EXTRA_TOOL_DIRS:
            resolved = shutil.which(argv[0], path=extra)
            if resolved is not None:
                break
    if resolved is None:
        return None
    try:
        completed = subprocess.run(
            [resolved, *argv[1:]], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return -1, "", f"{type(exc).__name__}: {exc}"
    return completed.returncode, completed.stdout, completed.stderr


def check_nvidia_smi() -> dict:
    argv = [
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,compute_cap,memory.total",
        "--format=csv,noheader",
    ]
    command = " ".join(argv)
    result = _run_command(argv)
    if result is None:
        return _check(
            "nvidia_smi", UNAVAILABLE, {"reason": "nvidia-smi not found in PATH"}, command
        )
    code, out, err = result
    if code != 0:
        return _check(
            "nvidia_smi", FAIL, {"exit_code": code, "stderr": err.strip()[-400:]}, command
        )
    gpus = []
    for line in out.strip().splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            return _check(
                "nvidia_smi", FAIL, {"reason": "unexpected csv shape", "line": line}, command
            )
        gpus.append(
            {
                "name": fields[0],
                "uuid": fields[1],
                "driver_version": fields[2],
                "compute_capability": fields[3],
                "memory_total": fields[4],
            }
        )
    if not gpus:
        return _check("nvidia_smi", FAIL, {"reason": "no gpu rows in output"}, command)
    return _check("nvidia_smi", PASS, {"gpus": gpus}, command)


def check_nvcc() -> dict:
    result = _run_command(["nvcc", "--version"])
    if result is None:
        return _check("nvcc_toolchain", UNAVAILABLE, {"reason": "nvcc not found in PATH"})
    code, out, _err = result
    if code != 0:
        return _check("nvcc_toolchain", FAIL, {"exit_code": code})
    match = re.search(r"release ([0-9.]+), V([0-9.]+)", out)
    if match is None:
        return _check("nvcc_toolchain", FAIL, {"reason": "unparseable nvcc --version output"})
    return _check("nvcc_toolchain", PASS, {"release": match.group(1), "version": match.group(2)})


def check_ncu() -> dict:
    result = _run_command(["ncu", "--version"])
    if result is None:
        return _check("ncu_profiler", UNAVAILABLE, {"reason": "ncu not found in PATH"})
    code, out, err = result
    if code != 0:
        return _check("ncu_profiler", FAIL, {"exit_code": code, "stderr": err.strip()[-400:]})
    match = re.search(r"Version ([0-9.]+)", out)
    version = match.group(1) if match else ""
    return _check(
        "ncu_profiler",
        PASS,
        {"version": version, "scope": "presence_and_version_only; real profiling is T15"},
    )


def check_optional_python_packages() -> dict:
    """torch/triton presence is informational: absent packages are normal."""
    detail: dict[str, str] = {}
    for module in ("torch", "triton"):
        try:
            imported = __import__(module)
            detail[module] = getattr(imported, "__version__", "unknown")
        except Exception:  # noqa: BLE001 - absence is the expected common case
            detail[module] = "not installed"
    return _check("optional_python_packages", PASS, detail)


def _load_cuda_driver_lib() -> ctypes.CDLL | None:
    if sys.platform == "win32":
        try:
            return ctypes.WinDLL("nvcuda.dll")
        except OSError:
            return None
    for name in ("libcuda.so.1", "libcuda.so"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


_CUdeviceptr = ctypes.c_uint64 if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_uint32


_ARGTYPES = {
    "cuInit": [ctypes.c_uint],
    "cuDeviceGetCount": [ctypes.POINTER(ctypes.c_int)],
    "cuDeviceGet": [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
    "cuDeviceGetName": [ctypes.c_char_p, ctypes.c_int, ctypes.c_int],
    "cuDeviceComputeCapability": [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ],
    "cuDevicePrimaryCtxRetain": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int],
    "cuDevicePrimaryCtxRelease_v2": [ctypes.c_int],
    "cuCtxSetCurrent": [ctypes.c_void_p],
    "cuModuleLoadData": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p],
    "cuModuleGetFunction": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p],
    "cuMemAlloc_v2": [ctypes.POINTER(_CUdeviceptr), ctypes.c_size_t],
    "cuMemAllocManaged": [ctypes.POINTER(_CUdeviceptr), ctypes.c_size_t, ctypes.c_uint],
    "cuMemFree_v2": [_CUdeviceptr],
    "cuMemcpyDtoH_v2": [ctypes.c_void_p, _CUdeviceptr, ctypes.c_size_t],
    "cuLaunchKernel": [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ],
    "cuCtxSynchronize": [],
    "cuModuleUnload": [ctypes.c_void_p],
}


def _configure_argtypes(lib: ctypes.CDLL) -> None:
    """Declare prototypes so integers widen correctly: without argtypes,
    ctypes passes Python ints as 32-bit values and 64-bit size_t parameters
    receive garbage in their upper half (observed as cuMemAlloc failures).
    Plain Python doubles (tests) cannot carry argtypes; skip those."""
    for name, prototype in _ARGTYPES.items():
        fn = getattr(lib, name, None)
        if fn is None:
            continue
        try:
            fn.argtypes = prototype
        except AttributeError:
            continue


def _driver_api_sequence(lib: ctypes.CDLL) -> dict:
    _configure_argtypes(lib)
    try:
        result = ctypes.c_int(0)
        if lib.cuInit(0) != 0:
            return {"status": FAIL, "detail": {"reason": "cuInit failed"}}
        if lib.cuDeviceGetCount(ctypes.byref(result)) != 0:
            return {"status": FAIL, "detail": {"reason": "cuDeviceGetCount failed"}}
        if result.value < 1:
            return {
                "status": UNAVAILABLE,
                "detail": {"reason": "driver initialized but reports zero CUDA devices"},
            }
        device = ctypes.c_int(0)
        if lib.cuDeviceGet(ctypes.byref(device), 0) != 0:
            return {"status": FAIL, "detail": {"reason": "cuDeviceGet failed"}}
        name_buffer = ctypes.create_string_buffer(256)
        if lib.cuDeviceGetName(name_buffer, 256, device) != 0:
            return {"status": FAIL, "detail": {"reason": "cuDeviceGetName failed"}}
        device_name = name_buffer.value.decode("utf-8", errors="replace")
        major, minor = ctypes.c_int(0), ctypes.c_int(0)
        if lib.cuDeviceComputeCapability(ctypes.byref(major), ctypes.byref(minor), device) != 0:
            return {"status": FAIL, "detail": {"reason": "cuDeviceComputeCapability failed"}}
        return {
            "status": PASS,
            "detail": {
                "device_0": device_name,
                "compute_capability": f"{major.value}.{minor.value}",
                "device_count": result.value,
            },
        }
    except Exception as exc:  # noqa: BLE001 - report the concrete driver failure
        return {"status": FAIL, "detail": {"reason": f"{type(exc).__name__}: {exc}"}}


def _kernel_launch_sequence(lib: ctypes.CDLL) -> dict:
    """Small real probe: JIT an embedded PTX kernel, launch it, verify the
    device actually wrote 42. Device memory is tried first; if that driver
    call fails, managed memory is tried as a separate supported allocation
    path. Every step records its CUresult so a failure is diagnosable."""
    _configure_argtypes(lib)
    trace: dict[str, str] = {}

    def step(name: str, curesult: int) -> bool:
        trace[name] = f"0x{curesult:x}"
        return curesult == 0

    try:
        if not step("cuInit", lib.cuInit(0)):
            return {"status": FAIL, "detail": {"reason": "cuInit failed", "trace": trace}}
        device = ctypes.c_int(0)
        if not step("cuDeviceGet", lib.cuDeviceGet(ctypes.byref(device), 0)):
            return {"status": FAIL, "detail": {"reason": "cuDeviceGet failed", "trace": trace}}
        context = ctypes.c_void_p()
        if not step(
            "cuDevicePrimaryCtxRetain", lib.cuDevicePrimaryCtxRetain(ctypes.byref(context), device)
        ):
            return {
                "status": FAIL,
                "detail": {"reason": "cuDevicePrimaryCtxRetain failed", "trace": trace},
            }
        try:
            if not step("cuCtxSetCurrent", lib.cuCtxSetCurrent(context)):
                return {
                    "status": FAIL,
                    "detail": {"reason": "cuCtxSetCurrent failed", "trace": trace},
                }
            module = ctypes.c_void_p()
            if not step(
                "cuModuleLoadData", lib.cuModuleLoadData(ctypes.byref(module), _PTX.encode("utf-8"))
            ):
                return {
                    "status": FAIL,
                    "detail": {"reason": "cuModuleLoadData (PTX JIT) failed", "trace": trace},
                }
            try:
                function = ctypes.c_void_p()
                if not step(
                    "cuModuleGetFunction",
                    lib.cuModuleGetFunction(ctypes.byref(function), module, b"probe_fill"),
                ):
                    return {
                        "status": FAIL,
                        "detail": {"reason": "cuModuleGetFunction failed", "trace": trace},
                    }
                buffer = _CUdeviceptr()
                memory = "device"
                if not step("cuMemAlloc_v2", lib.cuMemAlloc_v2(ctypes.byref(buffer), 4)):
                    memory = "managed"
                    if not step(
                        "cuMemAllocManaged", lib.cuMemAllocManaged(ctypes.byref(buffer), 4, 1)
                    ):
                        return {
                            "status": FAIL,
                            "detail": {
                                "reason": "both cuMemAlloc and cuMemAllocManaged failed",
                                "trace": trace,
                            },
                        }
                try:
                    # kernelParams is an array of pointers to host-side argument
                    # storage. The kernel argument value is the CUdeviceptr, so
                    # passing buffer.value directly would lose one indirection.
                    kernel_arg = _CUdeviceptr(buffer.value)
                    params = (ctypes.c_void_p * 1)(
                        ctypes.cast(ctypes.byref(kernel_arg), ctypes.c_void_p)
                    )
                    launch_result = lib.cuLaunchKernel(
                        function, 1, 1, 1, 1, 1, 1, 0, None, params, None
                    )
                    if launch_result != 0:
                        return {
                            "status": FAIL,
                            "detail": {
                                "reason": f"cuLaunchKernel returned CUresult 0x{launch_result:x}",
                                "memory": memory,
                                "trace": trace,
                            },
                        }
                    step("cuLaunchKernel", launch_result)
                    sync_result = lib.cuCtxSynchronize()
                    if sync_result != 0:
                        return {
                            "status": FAIL,
                            "detail": {
                                "reason": f"cuCtxSynchronize returned CUresult "
                                f"0x{sync_result:x} after launch",
                                "memory": memory,
                                "trace": trace,
                            },
                        }
                    step("cuCtxSynchronize", sync_result)
                    host_value = ctypes.c_uint(0)
                    try:
                        read_ok = step(
                            "cuMemcpyDtoH_v2",
                            lib.cuMemcpyDtoH_v2(ctypes.byref(host_value), buffer, 4),
                        )
                    except OSError as exc:
                        # A driver call can raise an OS error instead of
                        # returning a CUresult; preserve that failure detail.
                        return {
                            "status": FAIL,
                            "detail": {
                                "reason": f"result read-back raised {type(exc).__name__}: {exc}",
                                "memory": memory,
                                "trace": trace,
                            },
                        }
                    if not read_ok:
                        return {
                            "status": FAIL,
                            "detail": {
                                "reason": "cuMemcpyDtoH_v2 failed",
                                "memory": memory,
                                "trace": trace,
                            },
                        }
                    if host_value.value != 42:
                        return {
                            "status": FAIL,
                            "detail": {
                                "reason": f"device wrote {host_value.value}, expected 42",
                                "memory": memory,
                                "trace": trace,
                            },
                        }
                finally:
                    step("cuMemFree_v2", lib.cuMemFree_v2(buffer))
            finally:
                step("cuModuleUnload", lib.cuModuleUnload(module))
        finally:
            step("cuDevicePrimaryCtxRelease_v2", lib.cuDevicePrimaryCtxRelease_v2(device))
    except Exception as exc:  # noqa: BLE001 - report the concrete failure
        return {
            "status": FAIL,
            "detail": {"reason": f"{type(exc).__name__}: {exc}", "trace": trace},
        }
    cleanup_failures = [
        name
        for name in ("cuMemFree_v2", "cuModuleUnload", "cuDevicePrimaryCtxRelease_v2")
        if trace.get(name) != "0x0"
    ]
    if cleanup_failures:
        return {
            "status": FAIL,
            "detail": {
                "reason": f"CUDA cleanup failed: {', '.join(cleanup_failures)}",
                "memory": memory,
                "trace": trace,
            },
        }
    return {
        "status": PASS,
        "detail": {
            "proof": f"PTX JIT kernel launched; device wrote 42 verified via {memory} round trip",
            "memory": memory,
            "trace": trace,
        },
    }


def _isolated_driver_check(mode: str) -> dict:
    """Run a driver check in a child process: driver libraries with broken
    driver call may terminate the process rather than return a CUresult. A
    crash in the child must degrade to an explicit failed check instead of
    killing the whole probe."""
    child = Path(__file__).resolve()
    try:
        completed = subprocess.run(
            [sys.executable, str(child), "--driver-child", mode],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": FAIL, "detail": {"reason": f"driver check child: {exc}"}}
    out = completed.stdout
    if _PROBE_MARKER in out:
        out = out.split(_PROBE_MARKER, 1)[1]
        try:
            parsed = json.loads(out)
        except json.JSONDecodeError:
            parsed = None
        if (
            isinstance(parsed, dict)
            and parsed.get("status") in (PASS, FAIL, UNAVAILABLE)
            and isinstance(parsed.get("detail"), dict)
        ):
            if completed.returncode == 0 or parsed["status"] == FAIL:
                return parsed
            return {
                "status": FAIL,
                "detail": {
                    "reason": "driver check child returned nonzero while claiming "
                    f"{parsed['status']} (exit code {completed.returncode})"
                },
            }
    reason = (
        f"driver check child crashed (exit code {completed.returncode})"
        if completed.returncode != 0
        else "driver check child returned invalid structured output"
    )
    return {"status": FAIL, "detail": {"reason": reason}}


def check_driver_api() -> dict:
    result = _isolated_driver_check("driver_api")
    return _check("cuda_driver_api", result["status"], result["detail"])


def check_cuda_kernel_launch() -> dict:
    result = _isolated_driver_check("kernel_launch")
    return _check("cuda_kernel_launch", result["status"], result["detail"])


def _driver_child(mode: str) -> int:
    lib = _load_cuda_driver_lib()
    if lib is None:
        result = {
            "status": UNAVAILABLE,
            "detail": {"reason": "CUDA driver library not loadable (nvcuda.dll / libcuda.so.1)"},
        }
    elif mode == "driver_api":
        result = _driver_api_sequence(lib)
    else:
        result = _kernel_launch_sequence(lib)
    sys.stdout.write(_PROBE_MARKER)
    json.dump(result, sys.stdout)
    sys.stdout.write("\n")
    return 1 if result["status"] == FAIL else 0


ALL_CHECKS = (
    check_nvidia_smi,
    check_driver_api,
    check_cuda_kernel_launch,
    check_nvcc,
    check_ncu,
    check_optional_python_packages,
)


def build_report() -> dict:
    checks = [check() for check in ALL_CHECKS]
    summary = {
        status: sum(1 for entry in checks if entry["status"] == status)
        for status in (PASS, FAIL, UNAVAILABLE)
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "probe_protocol": PROBE_PROTOCOL,
        "generated_at": _now(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "python": platform.python_version(),
        "checks": checks,
        "summary": summary,
    }


def main() -> int:
    if "--driver-child" in sys.argv:
        mode = sys.argv[sys.argv.index("--driver-child") + 1]
        return _driver_child(mode)
    report = build_report()
    # Sentinel lets a transport (wsl.exe adds its own stdout noise) slice the
    # report out reliably.
    sys.stdout.write("KERNELAGENT_PROBE_JSON_BEGIN\n")
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 1 if report["summary"][FAIL] else 0


if __name__ == "__main__":
    sys.exit(main())
