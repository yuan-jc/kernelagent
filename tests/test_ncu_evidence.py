"""NCU evidence adapter (T15): parsing, catalog, and semantics.

Pure CPU. The real report generation runs in the ADR-0003 diagnostic
container via examples/ncu_evidence_smoke.py; these tests pin the
interpretation semantics on representative raw CSV."""

import pytest

from kernelagent.adapters.profiling import (
    MISSING,
    LaunchMetrics,
    MetricCatalog,
    associate_launches,
    detect_profiling_blocker,
    evidence_view,
    parse_raw_csv,
)

RAW_CSV = (
    '"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream",'
    '"Block Size","Grid Size","Device","CC","gpu__time_duration.sum","launch__registers_per_thread"\n'
    '"","","","","","","","","","","","us",""\n'
    '"0","108","two_kernels","127.0.0.1","saxpy(int, float, float *, float *)","1","7",'
    '"(256, 1, 1)","(4096, 1, 1)","0","8.9","12.5","32"\n'
    '"1","108","two_kernels","127.0.0.1","scale(int, float *)","1","7",'
    '"(256, 1, 1)","(4096, 1, 1)","0","8.9","3.2","16"\n'
    '"2","108","two_kernels","127.0.0.1","scale(int, float *)","1","7",'
    '"(256, 1, 1)","(4096, 1, 1)","0","8.9","3.1","16"\n'
)


def test_parse_extracts_all_launches_in_order():
    launches = parse_raw_csv(RAW_CSV)
    assert [launch.kernel_name.split("(")[0] for launch in launches] == ["saxpy", "scale", "scale"]
    assert [launch.launch_id for launch in launches] == ["0", "1", "2"]


def test_launch_metrics_units_and_values():
    launches = parse_raw_csv(RAW_CSV)
    duration = launches[0].metrics["gpu__time_duration.sum"]
    assert duration == {"unit": "us", "value": 12.5}
    assert launches[0].compute_capability == "8.9"
    assert launches[0].grid_size == "(4096, 1, 1)"


def test_missing_metric_is_missing_never_zero():
    launches = parse_raw_csv(RAW_CSV)
    assert launches[0].value("dram__throughput.avg.pct_of_peak_sustained_elapsed") is MISSING
    catalog = MetricCatalog.default()
    view = evidence_view(launches, catalog)
    assert "dram__throughput.avg.pct_of_peak_sustained_elapsed" in (
        view["catalog_metrics_missing_in_all_launches"]
    )
    assert view["note"].startswith("missing metrics are absent")


def test_zero_is_distinguishable_from_missing():
    launches = parse_raw_csv(RAW_CSV)
    assert launches[0].value("gpu__time_duration.sum") == 12.5
    assert launches[0].value("gpu__time_duration.sum") is not MISSING


def test_launch_association_keeps_all_and_order():
    launches = parse_raw_csv(RAW_CSV)
    scale_launches = associate_launches(launches, "scale")
    assert [launch.launch_id for launch in scale_launches] == ["1", "2"]
    assert len(associate_launches(launches, "saxpy")) == 1


def test_evidence_view_carries_profile_source_tag():
    view = evidence_view(parse_raw_csv(RAW_CSV), MetricCatalog.default())
    assert view["source"] == "ncu_profile"
    assert all(launch["source"] == "ncu_profile" for launch in view["launches"])


def test_blocker_signatures_map_to_explicit_states():
    assert (
        detect_profiling_blocker("==ERROR== ERR_NVGPUCTRPERM - The user does not have permission")
        == "gpu_counter_permission_denied"
    )
    assert detect_profiling_blocker("==WARNING== No kernels were profiled.") == "no_kernels_profiled"
    assert detect_profiling_blocker("all good") is None


def test_catalog_lookup_unknown_metric_is_none():
    catalog = MetricCatalog.default()
    assert catalog.unit_of("gpu__time_duration.sum") == "us"
    assert catalog.unit_of("not_a_metric") is None
    catalog.register("not_a_metric", "apples")
    assert catalog.unit_of("not_a_metric") == "apples"


def test_launch_metrics_dataclass_validates_via_value():
    launch = LaunchMetrics(
        launch_id="0",
        kernel_name="k",
        grid_size="(1)",
        block_size="(1)",
        device="0",
        compute_capability="8.9",
        metrics={"m": {"value": 1, "unit": None}},
    )
    assert launch.value("m") == 1
    assert launch.value("absent") is MISSING
