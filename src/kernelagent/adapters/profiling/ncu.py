"""NCU evidence adapter (T15, design §10.2).

Two strictly separated halves:

- Generation: a real ``.ncu-rep`` is produced inside a *privileged
  diagnostic container* running only parent-authored fixed binaries
  (ADR-0003). This is a scoped exception: untrusted candidates are never
  profiled here, and their evaluation boundary (ADR-0001) is untouched.
  On the current host the alternative (non-root or plain-container
  profiling) is blocked by NVreg counter permissions - observed as
  ERR_NVGPUCTRPERM and recorded, never silently swallowed.

- Interpretation: ``ncu --import --csv --page raw`` runs on the host
  without GPU or elevated permissions. The parsed launches carry their
  metrics verbatim; a metric that was not collected is reported MISSING,
  never zero-filled. The result is tagged ``source="ncu_profile"`` so it
  can never be confused with formal timing (T07) results."""

import csv
import io
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

SOURCE_TAG = "ncu_profile"
MISSING = object()

_CATEGORIES: dict[str, str] = {
    "gpu__time_duration.sum": "us",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "%",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "%",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "%",
    "launch__grid_size": "blocks",
    "launch__block_size": "threads",
    "launch__registers_per_thread": "registers",
    "launch__occupancy_limit_registers": "%",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "%",
}


@dataclass(frozen=True, slots=True)
class MetricCatalog:
    """Metric name -> unit registry with an honest lookup: a metric absent
    from a launch is MISSING, never zero."""

    entries: dict[str, str]

    @classmethod
    def default(cls) -> "MetricCatalog":
        return cls(entries=dict(_CATEGORIES))

    def unit_of(self, metric: str) -> str | None:
        return self.entries.get(metric)

    def register(self, metric: str, unit: str) -> None:
        self.entries[metric] = unit


@dataclass(frozen=True, slots=True)
class LaunchMetrics:
    launch_id: str
    kernel_name: str
    grid_size: str
    block_size: str
    device: str
    compute_capability: str
    metrics: dict

    def entry(self, metric: str) -> dict | None:
        """Full metric entry (value + unit) or None when not collected."""
        entry = self.metrics.get(metric)
        return dict(entry) if isinstance(entry, dict) else None

    def value(self, metric: str):
        """Value for ``metric``, or MISSING when not collected - callers
        must distinguish MISSING from 0. Units live in ``entry()`` and the
        catalog."""
        entry = self.metrics.get(metric)
        return entry["value"] if isinstance(entry, dict) else MISSING

    def with_source(self) -> dict:
        """Serializable view tagged with its evidence source."""
        return {
            "source": SOURCE_TAG,
            "launch_id": self.launch_id,
            "kernel_name": self.kernel_name,
            "grid_size": self.grid_size,
            "block_size": self.block_size,
            "device": self.device,
            "compute_capability": self.compute_capability,
            "metrics": self.metrics,
        }


_SKIP_COLUMNS = frozenset({"ID", "Kernel Name", "Grid Size", "Block Size", "Device", "CC"})


def parse_raw_csv(csv_text: str) -> list[LaunchMetrics]:
    """Parse ``ncu --import X --csv --page raw`` output: row 0 = metric
    names, row 1 = units, rows 2+ = one launch each."""
    reader = csv.reader(io.StringIO(csv_text))
    rows = [row for row in reader if row]
    if len(rows) < 3:
        raise ValueError("raw CSV needs a header row, a units row and at least one launch row")
    header = rows[0]

    def col(name: str) -> int | None:
        return header.index(name) if name in header else None

    units = rows[1]
    id_i = col("ID")
    kernel_i = col("Kernel Name")
    grid_i = col("Grid Size")
    block_i = col("Block Size")
    device_i = col("Device")
    cc_i = col("CC")
    if id_i is None or kernel_i is None:
        raise ValueError("raw CSV is missing ID / Kernel Name columns")

    launches: list[LaunchMetrics] = []
    for row in rows[2:]:
        metrics: dict[str, dict] = {}
        for index, name in enumerate(header):
            if index >= len(row) or name in _SKIP_COLUMNS:
                continue
            raw = row[index].strip()
            if raw == "":
                continue
            unit = units[index].strip() if index < len(units) else ""
            entry: dict = {"unit": unit or None}
            try:
                entry["value"] = float(raw.replace(",", ""))
                if entry["value"] == int(entry["value"]) and "." not in raw:
                    entry["value"] = int(entry["value"])
            except ValueError:
                entry["value"] = raw
            metrics[name] = entry
        launches.append(
            LaunchMetrics(
                launch_id=row[id_i],
                kernel_name=row[kernel_i],
                grid_size=row[grid_i] if grid_i is not None and grid_i < len(row) else "",
                block_size=row[block_i] if block_i is not None and block_i < len(row) else "",
                device=row[device_i] if device_i is not None and device_i < len(row) else "",
                compute_capability=row[cc_i] if cc_i is not None and cc_i < len(row) else "",
                metrics=metrics,
            )
        )
    return launches


def associate_launches(launches: list[LaunchMetrics], kernel_name: str) -> list[LaunchMetrics]:
    """All launches of one kernel, in recorded order - profiling evidence
    must never collapse to 'the last kernel'."""
    return [launch for launch in launches if launch.kernel_name.startswith(kernel_name)]


def detect_profiling_blocker(stderr_text: str) -> str | None:
    """Map known NCU failure signatures to explicit states."""
    if "ERR_NVGPUCTRPERM" in stderr_text:
        return "gpu_counter_permission_denied"
    if "no kernels were profiled" in stderr_text.lower():
        return "no_kernels_profiled"
    if "INSUFFICIENT_DRIVER" in stderr_text:
        return "driver_insufficient"
    return None


def import_report(ncu_bin: str, report_path: Path, timeout: float = 120.0) -> list[LaunchMetrics]:
    """Read a real .ncu-rep on the host: import needs no GPU and no
    elevated permissions."""
    completed = subprocess.run(
        [ncu_bin, "--import", str(report_path), "--csv", "--page", "raw"],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"ncu --import failed rc={completed.returncode}: {completed.stderr[-300:]}"
        )
    return parse_raw_csv(completed.stdout)


def profile_in_diagnostic_container(
    *,
    binary_path: Path,
    output_path: Path,
    gpu_device: str,
    toolkit_root: Path = Path("/home/y/toolchains/cuda"),
    image: str = "kernelagent-eval:cuda124-pytorch251-t05",
    ncu_container_path: str = "/cuda/bin/ncu",
    set_name: str = "basic",
    docker_command: tuple[str, ...] = ("docker",),
) -> Path:
    """ADR-0003 diagnostic lease: profile a parent-authored fixed binary in
    a privileged container and copy the .ncu-rep back to ``output_path``.

    Scope guard: callers must only pass binaries they built themselves;
    untrusted candidates never enter this path (ADR-0001 governs them)."""
    binary_path = Path(binary_path).resolve()
    output_path = Path(output_path).resolve()
    workdir = output_path.parent
    workdir.mkdir(parents=True, exist_ok=True)
    script = (
        f"mkdir -p /work && cp {binary_path} /work/ && cd /work\n"
        f"{ncu_container_path} --set {set_name} -o report -f ./{binary_path.name}\n"
        f"cp report.ncu-rep {output_path}\n"
    )
    completed = subprocess.run(
        [
            *docker_command,
            "run",
            "--rm",
            "--privileged",
            "--network",
            "none",
            "--device",
            gpu_device,
            "-v",
            f"{Path(toolkit_root).resolve()}:/cuda:ro",
            "-v",
            f"{binary_path.parent}:/indata:ro",
            "-v",
            f"{workdir}:/out",
            image,
            "bash",
            "-c",
            script.replace(str(binary_path), f"/indata/{binary_path.name}").replace(
                str(output_path), "/out/" + output_path.name
            ),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    blocker = detect_profiling_blocker(completed.stderr + completed.stdout)
    if blocker:
        raise RuntimeError(f"profiling blocked: {blocker}")
    if not output_path.is_file():
        raise RuntimeError(
            f"report not produced (rc={completed.returncode}): {completed.stderr[-300:]}"
        )
    return output_path


def evidence_view(launches: list[LaunchMetrics], catalog: MetricCatalog) -> dict:
    """Serializable evidence: every launch verbatim plus catalog units and
    explicit missing-metric accounting."""
    catalogued = sorted(catalog.entries)
    missing = []
    for metric in catalogued:
        if all(launch.value(metric) is MISSING for launch in launches):
            missing.append(metric)
    return {
        "source": SOURCE_TAG,
        "launch_count": len(launches),
        "launches": [launch.with_source() for launch in launches],
        "catalog": dict(catalog.entries),
        "catalog_metrics_missing_in_all_launches": missing,
        "note": "missing metrics are absent, never zero-filled",
    }


def to_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str)
