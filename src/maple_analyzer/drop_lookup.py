"""Map-to-monster drop lookup backed by the public MapleMemory database.

The website publishes two JSON assignments rather than a small JSON API:
``maps-data.js`` contains map spawn records and ``data.js`` contains monster
drop records.  This module keeps the parsing and matching independent from
Tkinter so a slow/unavailable network never blocks the live OCR loop and the
matching rules can be regression-tested with small fixtures.
"""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import json
import re
import threading
import unicodedata
from typing import Any, Mapping
from urllib.parse import quote
from urllib.request import Request, urlopen


SITE_BASE_URL = "https://morrisrrrrrrr-svg.github.io/"
MAPS_DATA_URL = f"{SITE_BASE_URL}maps-data.js"
DROP_DATA_URL = f"{SITE_BASE_URL}data.js"
MAPS_PAGE_URL = f"{SITE_BASE_URL}maps.html"

_USER_AGENT = "MapleStoryAnalyzer/0.4 (map drop lookup)"
_REMOTE_CACHE: tuple[Mapping[str, Any], Mapping[str, Any]] | None = None
_CACHE_LOCK = threading.Lock()
_BARRACKS_MAP_RE = re.compile(r"第\d+軍營")
# The tiny mini-map font commonly turns 營 into 管/营/営.  Keep this alias
# deliberately narrow: replacing 管 globally would turn unrelated map names
# into a different map, but it is safe when the surrounding token is the
# distinctive numbered barracks family.
_BARRACKS_OCR_RE = re.compile(r"第(\d+)軍(?:營|管|营|営)")
_ROMAN_SUFFIX_RE = re.compile(r"(IV|III|II|I|V|X)+$", re.IGNORECASE)


class DropLookupError(RuntimeError):
    """Raised when the public database cannot be downloaded or decoded."""


@dataclass(frozen=True)
class DropItem:
    item_id: str
    name: str
    category: str
    subcategory: str
    probability: float | None
    min_quantity: int | None
    max_quantity: int | None
    source_label: str
    source_url: str
    description: str


@dataclass(frozen=True)
class MonsterDropSummary:
    monster_id: str
    name: str
    level: int | None
    spawn_count: int
    drops: tuple[DropItem, ...]

    @property
    def source_url(self) -> str:
        return monster_page_url(self.monster_id)


@dataclass(frozen=True)
class MapDropSummary:
    map_id: str
    map_name: str
    map_label: str
    monsters: tuple[MonsterDropSummary, ...]
    generated_at: str | None

    @property
    def source_url(self) -> str:
        return map_page_url(self.map_id)


def normalize_map_name(value: object) -> str:
    """Normalize OCR/user-entered map text without changing Chinese names."""
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    # OCR can return a BOM/zero-width direction mark around the tiny mini-map
    # label.  These are invisible in the UI but make an exact map lookup miss.
    text = "".join(
        character for character in text
        if unicodedata.category(character) not in {"Cc", "Cf"}
    )
    text = text.translate(str.maketrans({"军": "軍", "营": "營"}))
    text = re.sub(r"^(?:小地圖|小地图|地圖|地图)[:：]?", "", text)
    text = re.sub(r"[\s\-_·•/\\]+", "", text).casefold()
    # Normalize only the known OCR-confusion pattern.  This lets the live
    # reader resolve e.g. 第3軍管 to the canonical 第3軍營 record without
    # weakening matching for every other Chinese map name.
    text = _BARRACKS_OCR_RE.sub(lambda match: f"第{match.group(1)}軍營", text)
    return text


def parse_database_script(payload: str, variable_name: str) -> Mapping[str, Any]:
    """Decode ``window.<variable_name> = {...};`` from a site script."""
    marker = f"window.{variable_name}"
    marker_index = payload.find(marker)
    if marker_index < 0:
        raise DropLookupError(f"database variable not found: {variable_name}")
    equals_index = payload.find("=", marker_index + len(marker))
    if equals_index < 0:
        raise DropLookupError(f"database assignment not found: {variable_name}")
    encoded = payload[equals_index + 1 :].strip()
    if encoded.endswith(";"):
        encoded = encoded[:-1].rstrip()
    try:
        decoded = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise DropLookupError(f"invalid database JSON: {variable_name}") from exc
    if not isinstance(decoded, Mapping):
        raise DropLookupError(f"database root is not an object: {variable_name}")
    return decoded


def _download_text(url: str, timeout: float) -> str:
    request = Request(url, headers={"User-Agent": _USER_AGENT, "Accept": "application/javascript"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except Exception as exc:  # urllib errors vary by Windows/Python version.
        raise DropLookupError(f"download failed: {url}") from exc


def clear_remote_cache() -> None:
    """Clear the process cache, primarily useful after a source update."""
    global _REMOTE_CACHE
    with _CACHE_LOCK:
        _REMOTE_CACHE = None


def load_remote_databases(
    *, timeout: float = 12.0, force_refresh: bool = False
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Load and cache the map and monster databases for this process."""
    global _REMOTE_CACHE
    with _CACHE_LOCK:
        if _REMOTE_CACHE is not None and not force_refresh:
            return _REMOTE_CACHE

    maps_db = parse_database_script(_download_text(MAPS_DATA_URL, timeout), "MS_MAP_DB")
    drops_db = parse_database_script(_download_text(DROP_DATA_URL, timeout), "MS_DROP_DB")
    with _CACHE_LOCK:
        _REMOTE_CACHE = (maps_db, drops_db)
    return maps_db, drops_db


def map_page_url(map_id: object) -> str:
    return f"{MAPS_PAGE_URL}?map={quote(str(map_id), safe='')}"


def monster_page_url(monster_id: object) -> str:
    return f"{SITE_BASE_URL}?monster={quote(str(monster_id), safe='')}"


def _text(record: Mapping[str, Any], key: str, default: str = "") -> str:
    value = record.get(key)
    return str(value).strip() if value is not None else default


def _number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _integer(value: object) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def _choose_drop_rate(drop: Mapping[str, Any]) -> tuple[float | None, Mapping[str, Any]]:
    rates = [rate for rate in drop.get("dropRates", ()) if isinstance(rate, Mapping)]
    if not rates:
        return None, {}

    # Prefer a rate with an actual probability.  Some rows contain only a
    # source label, while another source row for the same item has a usable
    # value.  Keep the first usable row deterministic.
    for rate in rates:
        probability = _number(rate.get("probability"))
        if probability is None:
            probability = _number(rate.get("probabilityApprox"))
        if probability is not None:
            return probability, rate
    return None, rates[0]


def _drop_item(drop: Mapping[str, Any]) -> DropItem:
    probability, rate = _choose_drop_rate(drop)
    minimum = _integer(rate.get("min"))
    maximum = _integer(rate.get("max"))
    if minimum is None:
        minimum = _integer(drop.get("min"))
    if maximum is None:
        maximum = _integer(drop.get("max"))
    return DropItem(
        item_id=_text(drop, "id"),
        name=_text(drop, "name") or _text(drop, "id", "Unknown item"),
        category=_text(drop, "category") or _text(drop, "kind"),
        subcategory=_text(drop, "subcategory"),
        probability=probability,
        min_quantity=minimum,
        max_quantity=maximum,
        source_label=_text(rate, "sourceLabel") or _text(rate, "source") or "",
        source_url=_text(rate, "sourceUrl"),
        description=_text(drop, "desc"),
    )


def _map_match_score(query: str, candidate: str, field: str) -> int:
    if not query or not candidate:
        return -1
    if candidate == query:
        return {"name": 100, "label": 95, "areaName": 90}.get(field, 80)
    if candidate.endswith(query) or query.endswith(candidate):
        return {"name": 80, "label": 75, "areaName": 70}.get(field, 60)
    if query in candidate or candidate in query:
        return {"name": 55, "label": 50, "areaName": 45}.get(field, 40)

    # The same OCR model can confuse the leading glyph or 墓/基, while keeping
    # the distinctive "之墓" family and Roman floor suffix.  That structure
    # is safe to use as a stronger fallback because the floor suffix separates
    # 遺跡之墓Ⅰ/Ⅱ/Ⅲ/Ⅳ from one another.
    if field == "name":
        query_family = _map_family_signature(query)
        candidate_family = _map_family_signature(candidate)
        if query_family and query_family == candidate_family:
            return 64

    # Tiny mini-map labels are prone to dropping one Chinese glyph (for
    # example, OCR may return 遺之墓IV for 遺跡之墓Ⅳ).  A conservative fuzzy
    # fallback keeps exact/substring matching authoritative and only accepts a
    # close match when the Roman map-floor suffix agrees.  This is deliberately
    # not a broad fuzzy search: a wrong map is worse than a temporary miss.
    if field == "name" and len(query) >= 5 and len(candidate) >= 5:
        query_suffix = _ROMAN_SUFFIX_RE.search(query)
        candidate_suffix = _ROMAN_SUFFIX_RE.search(candidate)
        if query_suffix and candidate_suffix and query_suffix.group(0) != candidate_suffix.group(0):
            return -1
        ratio = SequenceMatcher(None, query, candidate, autojunk=False).ratio()
        if ratio >= 0.78:
            return 32 + round(ratio * 20)
    return -1


def _map_family_signature(value: str) -> str | None:
    """Return a stable signature for the numbered 遺跡之墓 map family."""
    suffix = _ROMAN_SUFFIX_RE.search(value)
    if suffix is None:
        return None
    core = value[:suffix.start()]
    if core.endswith("之墓") or core.endswith("之基"):
        return f"之墓{suffix.group(0).casefold()}"
    return None


def _find_map(map_name: str, maps_db: Mapping[str, Any]) -> Mapping[str, Any] | None:
    query = normalize_map_name(map_name)
    # When the mini-map crop includes a region prefix or a stray OCR suffix,
    # the stable "第N軍營" token is still enough to choose the canonical map.
    # Keep the regular query as a fallback for maps outside this naming family.
    queries = [query]
    barracks_match = _BARRACKS_MAP_RE.search(query)
    if barracks_match and barracks_match.group(0) not in queries:
        queries.insert(0, barracks_match.group(0))
    best: tuple[int, Mapping[str, Any]] | None = None
    for record in maps_db.get("maps", ()):
        if not isinstance(record, Mapping):
            continue
        for candidate_query in queries:
            for field in ("name", "label", "areaName"):
                score = _map_match_score(candidate_query, normalize_map_name(record.get(field)), field)
                if score < 0:
                    continue
                if best is None or score > best[0]:
                    best = score, record
    return best[1] if best is not None else None


def lookup_map_drops(
    map_name: str,
    maps_db: Mapping[str, Any],
    drops_db: Mapping[str, Any],
) -> MapDropSummary | None:
    """Resolve one OCR map name to its unique spawned monsters and drops."""
    map_record = _find_map(map_name, maps_db)
    if map_record is None:
        return None

    monsters_by_id = {
        _text(record, "id"): record
        for record in drops_db.get("monsters", ())
        if isinstance(record, Mapping) and _text(record, "id")
    }
    spawn_ids: dict[str, dict[str, Any]] = {}
    for spawn in map_record.get("monsterSpawns", ()):
        if not isinstance(spawn, Mapping):
            continue
        monster_id = _text(spawn, "monsterId") or _text(spawn, "id")
        if not monster_id:
            continue
        entry = spawn_ids.setdefault(
            monster_id,
            {"name": _text(spawn, "name") or monster_id, "level": _integer(spawn.get("level")), "count": 0},
        )
        entry["count"] += 1

    monsters: list[MonsterDropSummary] = []
    for monster_id, spawn in spawn_ids.items():
        record = monsters_by_id.get(monster_id)
        if record is None:
            monsters.append(MonsterDropSummary(
                monster_id=monster_id,
                name=spawn["name"],
                level=spawn["level"],
                spawn_count=spawn["count"],
                drops=(),
            ))
            continue
        drops = [
            _drop_item(drop)
            for drop in record.get("drops", ())
            if isinstance(drop, Mapping)
        ]
        drops.sort(key=lambda item: (item.category, item.subcategory, item.name))
        monsters.append(MonsterDropSummary(
            monster_id=monster_id,
            name=_text(record, "name") or spawn["name"],
            level=_integer(record.get("level")) if record.get("level") is not None else spawn["level"],
            spawn_count=spawn["count"],
            drops=tuple(drops),
        ))

    metadata = maps_db.get("metadata")
    generated_at = _text(metadata, "generatedAt") if isinstance(metadata, Mapping) else None
    return MapDropSummary(
        map_id=_text(map_record, "id"),
        map_name=_text(map_record, "name") or map_name,
        map_label=_text(map_record, "label") or _text(map_record, "name") or map_name,
        monsters=tuple(monsters),
        generated_at=generated_at or None,
    )


def fetch_map_drop_summary(map_name: str, *, timeout: float = 12.0) -> MapDropSummary | None:
    maps_db, drops_db = load_remote_databases(timeout=timeout)
    return lookup_map_drops(map_name, maps_db, drops_db)


def format_probability(probability: float | None) -> str:
    """Format a probability as a compact percentage without inventing data."""
    if probability is None or probability < 0:
        return "—"
    percentage = probability * 100
    if percentage == 0:
        return "0%"
    if percentage >= 1:
        return f"{percentage:.2f}%"
    if percentage >= 0.1:
        return f"{percentage:.3f}%"
    return f"{percentage:.4f}%"


def format_quantity(minimum: int | None, maximum: int | None) -> str:
    if minimum is None and maximum is None:
        return ""
    if minimum is None:
        return f"×{maximum}"
    if maximum is None or minimum == maximum:
        return f"×{minimum}"
    return f"×{minimum}–{maximum}"
