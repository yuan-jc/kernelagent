"""Profiling adapters (T15 + baseline pipeline): NCU evidence generation
(privileged diagnostic lease, ADR-0003) and interpretation. Profiling
evidence is source-tagged and never merges with T07 timing results."""

from kernelagent.adapters.profiling.ncu import (
    MISSING,
    SOURCE_TAG,
    LaunchMetrics,
    MetricCatalog,
    associate_launches,
    detect_profiling_blocker,
    evidence_view,
    import_report,
    parse_raw_csv,
    profile_in_diagnostic_container,
)
from kernelagent.adapters.profiling.pipeline import (
    DEFAULT_METRIC_LIST,
    ProfileCapture,
    build_baseline_profiler,
    host_ncu_bin,
    load_collected_evidence,
    profile_baseline,
    stage_profile_inputs,
    summarize_evidence,
)

__all__ = [
    "DEFAULT_METRIC_LIST",
    "MISSING",
    "SOURCE_TAG",
    "LaunchMetrics",
    "MetricCatalog",
    "ProfileCapture",
    "associate_launches",
    "build_baseline_profiler",
    "detect_profiling_blocker",
    "evidence_view",
    "host_ncu_bin",
    "import_report",
    "load_collected_evidence",
    "parse_raw_csv",
    "profile_baseline",
    "profile_in_diagnostic_container",
    "stage_profile_inputs",
    "summarize_evidence",
]
