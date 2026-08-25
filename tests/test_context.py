"""Low-frequency map/job context extraction tests."""
from __future__ import annotations

from PIL import Image

from maple_analyzer.monitor import extract_context


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
