"""Profiling adapters (T15): NCU evidence generation (privileged
diagnostic lease, ADR-0003) and interpretation. Profiling evidence is
source-tagged and never merges with T07 timing results."""

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

__all__ = [
    "MISSING",
    "SOURCE_TAG",
    "LaunchMetrics",
    "MetricCatalog",
    "associate_launches",
    "detect_profiling_blocker",
    "evidence_view",
    "import_report",
    "parse_raw_csv",
    "profile_in_diagnostic_container",
]
