"""Low-frequency map/job context extraction tests."""
from __future__ import annotations

from PIL import Image

from maple_analyzer.monitor import extract_context
from maple_analyzer.overlay import OverlayApp
from maple_analyzer.regions import CONTEXT_BOXES


def test_reference_map_boxes_cover_the_second_mini_map_text_row():
    # In the 1351x800 reference client, the first map text row ends before
    # y=75 and the actual map title occupies the second row below it.
    assert CONTEXT_BOXES["map"] == (0, 75, 145, 101)
    assert CONTEXT_BOXES["map_wide"] == (0, 70, 200, 110)


class _ContextOcr:
    """Return the strings seen in the real screenshot at each crop scale."""

    def read_field(self, image):
        # The map's enlarged second-line crop is 600x160 in the reference
        # client.  The enlarged LV/job crop is wider and taller.
        if image.height == 160:
            return "第3军营"
        if image.height >= 200:
            return "快咨 6 8 LV."
        return ""

    def read_lines(self, _image):
        return []


def test_context_reads_the_second_map_line_and_normalizes_ocr_glyphs():
    reading = extract_context(
        _ContextOcr(),
        {
            "map": Image.new("RGB", (75, 20)),
            "job": Image.new("RGB", (172, 43)),
        },
    )

    assert reading.map_name == "第3軍營"
    assert reading.job_name == "俠盜"
    assert reading.map_confirmed is True
    assert reading.job_confirmed is True


class _WideRecoveryOcr:
    def read_field(self, image):
        # The focused crop only returns the final two glyphs.  The wider
        # retry recovers the identifying number and the full map token.
        if image.height == 160:
            return "軍/營"
        if image.height == 216:
            return "裝第3军营"
        if image.height >= 200:
            return "快咨 6 8 LV."
        return ""

    def read_lines(self, _image):
        return []


def test_context_prefers_complete_map_from_wider_retry():
    reading = extract_context(
        _WideRecoveryOcr(),
        {
            "map": Image.new("RGB", (75, 20)),
            "map_wide": Image.new("RGB", (110, 27)),
            "job": Image.new("RGB", (172, 43)),
        },
    )

    assert reading.map_name == "第3軍營"
    assert reading.map_confirmed is True


class _RomanFloorOcr:
    def read_field(self, image):
        # The focused view loses the trailing floor marker, while the wider
        # view retains the Unicode glyph rendered by the game.
        if image.height == 160:
            return "寺院通道"
        if image.height == 216:
            return "寺院通道Ⅱ"
        return ""

    def read_lines(self, _image):
        return []


def test_context_prefers_map_candidate_with_roman_floor_suffix():
    reading = extract_context(
        _RomanFloorOcr(),
        {
            "map": Image.new("RGB", (75, 20)),
            "map_wide": Image.new("RGB", (110, 27)),
        },
    )

    assert reading.map_name == "寺院通道II"
    assert reading.map_confirmed is True


def test_complete_floor_map_stops_after_native_probe_and_two_enlarged_views():
    class CountingOcr(_RomanFloorOcr):
        def __init__(self):
            self.field_calls = 0

        def read_field(self, image):
            self.field_calls += 1
            return super().read_field(image)

    ocr = CountingOcr()
    reading = extract_context(
        ocr,
        {
            "map": Image.new("RGB", (75, 20)),
            "map_wide": Image.new("RGB", (110, 27)),
        },
    )

    assert reading.map_name == "寺院通道II"
    assert ocr.field_calls == 3


def test_complete_floor_map_in_native_crop_skips_all_expensive_retries():
    class NativeFloorOcr:
        def __init__(self):
            self.field_calls = 0
            self.line_calls = 0

        def read_field(self, _image):
            self.field_calls += 1
            return "寺院通道Ⅱ"

        def read_lines(self, _image):
            self.line_calls += 1
            return []

    ocr = NativeFloorOcr()
    reading = extract_context(
        ocr,
        {
            "map": Image.new("RGB", (273, 49)),
            "map_wide": Image.new("RGB", (376, 68)),
        },
    )

    assert reading.map_name == "寺院通道II"
    assert reading.map_confirmed is True
    assert ocr.field_calls == 1
    assert ocr.line_calls == 0


def test_internally_confirmed_context_is_published_on_first_scan():
    app = OverlayApp.__new__(OverlayApp)

    app._accept_context_candidate("map", "寺院通道II", confirmed=True)

    assert app._detected_map_name == "寺院通道II"
    assert app._map_candidate_hits == 1


def test_unconfirmed_generic_context_still_requires_two_scans():
    app = OverlayApp.__new__(OverlayApp)

    app._accept_context_candidate("map", "神秘森林", confirmed=False)
    assert getattr(app, "_detected_map_name", None) is None
    app._accept_context_candidate("map", "神秘森林", confirmed=False)

    assert app._detected_map_name == "神秘森林"


def test_repeated_generic_map_views_in_one_scan_are_not_immediate_confirmation():
    """A foreground HUD label must not become a map from one OCR frame."""
    class GenericMapOcr:
        def read_field(self, _image):
            return "每60分鐘預估經驗"

        def read_lines(self, _image):
            return []

    reading = extract_context(
        GenericMapOcr(),
        {
            "map": Image.new("RGB", (75, 20)),
            "map_wide": Image.new("RGB", (110, 27)),
        },
    )

    assert reading.map_name == "每60分鐘預估經驗"
    assert reading.map_confirmed is False


class _WeakMapOnlyOcr:
    def read_field(self, _image):
        return "軍/營"

    def read_lines(self, _image):
        return ["軍/營"]


def test_context_does_not_publish_partial_barracks_name():
    reading = extract_context(
        _WeakMapOnlyOcr(),
        {
            "map": Image.new("RGB", (75, 20)),
            "map_wide": Image.new("RGB", (110, 27)),
            "job": Image.new("RGB", (172, 43)),
        },
    )

    assert reading.map_name is None


class _WeakMapFragmentOcr:
    def read_field(self, _image):
        return "軍管"

    def read_lines(self, _image):
        return ["軍管"]


def test_context_does_not_publish_bare_barracks_ocr_fragment():
    reading = extract_context(
        _WeakMapFragmentOcr(),
        {
            "map": Image.new("RGB", (75, 20)),
            "map_wide": Image.new("RGB", (110, 27)),
            "job": Image.new("RGB", (172, 43)),
        },
    )

    assert reading.map_name is None
