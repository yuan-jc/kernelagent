"""Offline method catalog (A2 research, generator-design §4, new-vs-extend).

The knowledge source is ``research/optimization-methods/methods.yaml``
(38 method entries). The runtime package has ZERO third-party dependencies
(enforced by tests/test_package.py), so the catalog ships as a frozen JSON
copy at ``generators/data/methods.json``. The copy records the source path
and the source file's sha256, so the freeze is verifiable without a YAML
parser (tests pin the hash against the research file).

Loading is strict and eager: any structural problem - unknown field, wrong
type, duplicate id, empty catalog - raises :class:`CatalogError` at load
time. A broken catalog is a hard configuration error, never a silently
empty directory (generator-design §4: "加载失败 → 显式错误").

The entries are hypothesis templates, not rules: ``expected_gain`` is a
ranking prior only, and every claim must be confirmed by the trusted
evaluator's formal timing/correctness results (design §0 trust boundary).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "methods.json"

SIGNAL_TIERS: tuple[str, ...] = ("tier_a_ncu", "tier_b_task", "tier_c_history")

# Categories present in the frozen v0.1 catalog. A new category is a catalog
# change and must extend this set deliberately (it drives ranking families).
KNOWN_CATEGORIES: frozenset[str] = frozenset(
    {
        "algorithmic",
        "compute_precision",
        "fusion",
        "hardware_target",
        "launch_overhead",
        "memory_layout",
        "numerics",
        "parallelism",
        "pipelining",
        "reduction",
        "resource_tuning",
        "search_discipline",
        "tuning_infra",
    }
)

_TOP_LEVEL_FIELDS = ("frozen_from", "frozen_source_sha256", "meta", "methods")
_META_FIELDS = ("version", "date", "hardware_target")


class CatalogError(ValueError):
    """Raised when the frozen method catalog fails structural validation."""


@dataclass(frozen=True, slots=True)
class MethodEntry:
    """One validated methods.yaml entry (hypothesis template, not a rule)."""

    id: str
    name: str
    category: str
    mechanism: str
    mechanism_detail: str
    applicability_signals: dict[str, tuple[str, ...]]
    expected_gain: str
    risks_preconditions: tuple[str, ...]
    triton_howto: tuple[str, ...]
    references: tuple[str, ...]

    def signals_for_tier(self, tier: str) -> tuple[str, ...]:
        return self.applicability_signals.get(tier, ())


@dataclass(frozen=True, slots=True)
class MethodCatalog:
    """Validated, immutable view of the frozen method catalog."""

    version: str
    date: str
    hardware_target: str
    frozen_from: str
    frozen_source_sha256: str
    methods: tuple[MethodEntry, ...]

    def __post_init__(self) -> None:
        if not self.methods:
            raise CatalogError("method catalog must not be empty")

    def __len__(self) -> int:
        return len(self.methods)

    def by_id(self) -> dict[str, MethodEntry]:
        return {entry.id: entry for entry in self.methods}


def _require_str_list(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise CatalogError(f"{field} must be a list of strings; got {value!r}")
    return tuple(value)


def _validate_method(raw: object, index: int) -> MethodEntry:
    where = f"methods[{index}]"
    if not isinstance(raw, dict):
        raise CatalogError(f"{where} must be a mapping; got {type(raw).__name__}")
    required = (
        "id",
        "name",
        "category",
        "mechanism",
        "mechanism_detail",
        "applicability_signals",
        "expected_gain",
        "risks_preconditions",
        "triton_howto",
        "references",
    )
    missing = [field for field in required if field not in raw]
    if missing:
        raise CatalogError(f"{where} is missing required fields: {missing}")
    unknown = sorted(set(raw) - set(required))
    if unknown:
        raise CatalogError(f"{where} has unknown fields: {unknown}")

    method_id = raw["id"]
    if not isinstance(method_id, str) or not method_id:
        raise CatalogError(f"{where}.id must be a non-empty string; got {method_id!r}")
    where = f"method {method_id!r}"
    for field in ("name", "category", "mechanism", "mechanism_detail", "expected_gain"):
        if not isinstance(raw[field], str) or not raw[field]:
            raise CatalogError(f"{where}.{field} must be a non-empty string; got {raw[field]!r}")
    if raw["category"] not in KNOWN_CATEGORIES:
        raise CatalogError(
            f"{where}.category {raw['category']!r} is not a known category "
            f"(extend generators.catalog.KNOWN_CATEGORIES deliberately)"
        )

    raw_signals = raw["applicability_signals"]
    if not isinstance(raw_signals, dict):
        raise CatalogError(f"{where}.applicability_signals must be a mapping; got {raw_signals!r}")
    unknown_tiers = sorted(set(raw_signals) - set(SIGNAL_TIERS))
    if unknown_tiers:
        raise CatalogError(f"{where}.applicability_signals has unknown tiers: {unknown_tiers}")
    if not any(_require_str_list_is_nonempty(raw_signals.get(tier)) for tier in SIGNAL_TIERS):
        raise CatalogError(f"{where}.applicability_signals must define at least one signal")
    signals = {
        tier: _require_str_list(raw_signals[tier], f"{where}.applicability_signals.{tier}")
        for tier in SIGNAL_TIERS
        if tier in raw_signals
    }
    return MethodEntry(
        id=method_id,
        name=raw["name"],
        category=raw["category"],
        mechanism=raw["mechanism"],
        mechanism_detail=raw["mechanism_detail"],
        applicability_signals=signals,
        expected_gain=raw["expected_gain"],
        risks_preconditions=_require_str_list(
            raw["risks_preconditions"], f"{where}.risks_preconditions"
        ),
        triton_howto=_require_str_list(raw["triton_howto"], f"{where}.triton_howto"),
        references=_require_str_list(raw["references"], f"{where}.references"),
    )


def _require_str_list_is_nonempty(value: object) -> bool:
    return isinstance(value, list) and len(value) > 0 and all(isinstance(i, str) for i in value)


def load_catalog(path: Path | str = DATA_PATH) -> MethodCatalog:
    """Parse and fully validate a frozen catalog JSON file."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogError(f"method catalog {path} could not be read: {exc}") from exc
    if not isinstance(payload, dict):
        raise CatalogError(f"method catalog {path} must be a JSON object")
    unknown = sorted(set(payload) - set(_TOP_LEVEL_FIELDS))
    if unknown:
        raise CatalogError(f"method catalog {path} has unknown top-level fields: {unknown}")
    for field in ("meta", "methods"):
        if field not in payload:
            raise CatalogError(f"method catalog {path} is missing required field {field!r}")

    meta = payload["meta"]
    if not isinstance(meta, dict):
        raise CatalogError(f"method catalog {path} meta must be an object")
    for field in _META_FIELDS:
        value = meta.get(field)
        if not isinstance(value, str) or not value:
            raise CatalogError(f"method catalog meta.{field} must be a non-empty string")

    raw_methods = payload["methods"]
    if not isinstance(raw_methods, list) or not raw_methods:
        raise CatalogError(f"method catalog {path} must contain a non-empty methods list")
    methods = tuple(_validate_method(raw, index) for index, raw in enumerate(raw_methods))
    ids = [entry.id for entry in methods]
    duplicates = sorted({method_id for method_id in ids if ids.count(method_id) > 1})
    if duplicates:
        raise CatalogError(f"method catalog has duplicate ids: {duplicates}")

    return MethodCatalog(
        version=meta["version"],
        date=meta["date"],
        hardware_target=meta["hardware_target"],
        frozen_from=str(payload.get("frozen_from", "")),
        frozen_source_sha256=str(payload.get("frozen_source_sha256", "")),
        methods=methods,
    )


def load_default() -> MethodCatalog:
    """Load the frozen catalog shipped inside the package."""
    return load_catalog(DATA_PATH)
