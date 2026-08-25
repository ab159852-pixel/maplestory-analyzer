from __future__ import annotations

from maple_analyzer.drop_lookup import (
    format_probability,
    format_quantity,
    lookup_map_drops,
    map_page_url,
    monster_page_url,
    normalize_map_name,
    parse_database_script,
)


def test_parse_database_script_reads_window_assignment():
    result = parse_database_script('window.MS_MAP_DB = {"maps": []};', "MS_MAP_DB")
    assert result == {"maps": []}


def test_third_barracks_resolves_unique_spawns_and_drop_rates():
    maps_db = {
        "metadata": {"generatedAt": "2026-08-25T02:16:10+08:00"},
        "maps": [{
            "id": "101030112",
            "name": "第3軍營",
            "label": "維多利亞 / 第3軍營",
            "monsterSpawns": [
                {"monsterId": "6230602", "name": "骷髏士官", "level": 63},
                {"monsterId": "6230602", "name": "骷髏士官", "level": 63},
                {"monsterId": "5150001", "name": "骷髏士兵", "level": 57},
            ],
        }],
    }
    drops_db = {"monsters": [
        {
            "id": 6230602,
            "name": "骷髏士官",
            "level": 63,
            "drops": [
                {
                    "id": 2000002,
                    "name": "白色藥水",
                    "category": "消耗",
                    "subcategory": "藥水",
                    "dropRates": [{
                        "probability": 0.015,
                        "sourceLabel": "MapleLandHub",
                        "sourceUrl": "https://example.invalid/rates",
                    }],
                },
                {
                    "id": 4000207,
                    "name": "骨盆",
                    "category": "其他",
                    "subcategory": "掉落道具",
                    "dropRates": [{"sourceLabel": "MapleLandHub"}],
                },
            ],
        },
        {"id": "5150001", "name": "骷髏士兵", "level": 57, "drops": []},
    ]}

    summary = lookup_map_drops("維多利亞 / 第3軍營", maps_db, drops_db)

    assert summary is not None
    assert summary.map_id == "101030112"
    assert summary.generated_at == "2026-08-25T02:16:10+08:00"
    assert [(monster.name, monster.spawn_count) for monster in summary.monsters] == [
        ("骷髏士官", 2),
        ("骷髏士兵", 1),
    ]
    officer_drops = summary.monsters[0].drops
    potion = next(item for item in officer_drops if item.name == "白色藥水")
    assert potion.probability == 0.015
    assert potion.source_label == "MapleLandHub"
    assert next(item for item in officer_drops if item.name == "骨盆").probability is None


def test_map_matching_handles_spaces_and_prefixes():
    maps_db = {"maps": [{"id": "1", "name": "第3軍營", "monsterSpawns": []}]}
    assert lookup_map_drops("地圖： 第 3 軍營", maps_db, {"monsters": []}) is not None
    # The tiny in-game map label frequently turns the final 營 glyph into 管.
    assert lookup_map_drops("第3軍管", maps_db, {"monsters": []}) is not None
    assert lookup_map_drops("\ufeff維多利亞 / 第3军营\u200b", maps_db, {"monsters": []}) is not None
    assert normalize_map_name("維多利亞 / 第3軍營") == "維多利亞第3軍營"
    assert lookup_map_drops("不存在的地圖", maps_db, {"monsters": []}) is None


def test_map_matching_recovers_one_missing_glyph_and_roman_floor():
    maps_db = {
        "maps": [
            {"id": "101030105", "name": "遺跡之墓Ⅰ", "monsterSpawns": []},
            {"id": "101030106", "name": "遺跡之墓Ⅱ", "monsterSpawns": []},
            {"id": "101030107", "name": "遺跡之墓Ⅲ", "monsterSpawns": []},
            {"id": "101030108", "name": "遺跡之墓Ⅳ", "monsterSpawns": []},
        ]
    }

    for ocr_text in ("遺之墓IV", "道之墓IV", "過之基IV"):
        summary = lookup_map_drops(ocr_text, maps_db, {"monsters": []})

        assert summary is not None, ocr_text
        assert summary.map_id == "101030108"
        assert summary.map_name == "遺跡之墓Ⅳ"


def test_display_helpers_and_source_links_are_stable():
    assert format_probability(None) == "—"
    assert format_probability(0.015) == "1.50%"
    assert format_probability(0.0003000003) == "0.0300%"
    assert format_quantity(None, None) == ""
    assert format_quantity(1, 1) == "×1"
    assert format_quantity(1, 3) == "×1–3"
    assert monster_page_url("6230602") == "https://morrisrrrrrrr-svg.github.io/?monster=6230602"
    assert map_page_url("101030112") == "https://morrisrrrrrrr-svg.github.io/maps.html?map=101030112"
