"""End-to-end golden-value regression: real capture crop -> real RapidOCR ->
real parser, against the one known-good screenshot in samples/. This is the
only test that exercises actual OCR inference (slow-ish, one-time model load)
-- it exists to catch regressions in FIELD_BOXES, the OCR wrapper, or the
parser regexes that unit tests on synthetic strings wouldn't catch, since
those don't touch real recognition output.

Ground truth verified 2026-08-17 by cropping and visually inspecting
samples/maple_story_ui.jpg (LV.44, HP[377/824], MP[1663/2816],
EXP162950[38.05%]) and confirming it matches this pipeline's output.
"""
import pytest

from maple_analyzer.capture import StaticImageCapture
from maple_analyzer import ocr as ocr_module
from maple_analyzer.ocr import OcrLine, StatPanelOcr
from maple_analyzer.parser import parse_fields

from conftest import SAMPLE_IMAGE


@pytest.fixture(scope="module")
def ocr_engine():
    return StatPanelOcr()  # loads the ONNX model once, reused across tests in this file


@pytest.fixture(scope="module")
def snapshot(ocr_engine):
    capture = StaticImageCapture(SAMPLE_IMAGE)
    fields = capture.grab_fields()
    field_text = {name: ocr_engine.read_field(img) for name, img in fields.items()}
    return parse_fields(field_text)


def test_level(snapshot):
    assert snapshot.level == 44


def test_hp(snapshot):
    assert (snapshot.hp_cur, snapshot.hp_max) == (377, 824)


def test_mp(snapshot):
    assert (snapshot.mp_cur, snapshot.mp_max) == (1663, 2816)


def test_exp(snapshot):
    assert snapshot.exp_cur == 162950
    assert snapshot.exp_pct == 38.05


def test_field_crops_are_nonempty(ocr_engine):
    capture = StaticImageCapture(SAMPLE_IMAGE)
    fields = capture.grab_fields()
    assert set(fields.keys()) == {"LV", "HP", "MP", "EXP"}
    for name, img in fields.items():
        assert img.width > 0 and img.height > 0, f"{name} crop is empty"


def test_shortcut_detection_splits_adjacent_quantity_runs_without_loading_ocr():
    ocr = StatPanelOcr.__new__(StatPanelOcr)
    ocr.read_lines = lambda _image: [
        OcrLine("Shift", y=21, x=74.75, left=50, right=100),
        OcrLine("1180.", y=54, x=73, left=44, right=102),
        OcrLine("2037465", y=54, x=167, left=110, right=224),
        OcrLine("3:03", y=117, x=264, left=243, right=287),
    ]

    from PIL import Image
    counts = ocr.read_shortcut_counts(Image.new("RGB", (297, 166)))

    assert counts["1"] == 1180
    assert counts["2"] == 2037
    assert counts["3"] == 465
    assert counts["8"] == 303


def test_blue_shortcut_prefers_complete_quantity_over_suffix_read():
    ocr = StatPanelOcr.__new__(StatPanelOcr)
    readings = iter([
        ("6", []),
        ("86", []),
        ("6", []),
        ("86", []),
    ])
    ocr._read_once = lambda _image: next(readings)

    from PIL import Image
    count = ocr_module._read_blue_shortcut_count(
        ocr,
        Image.new("RGB", (200, 200)),
        (997, 700, 1034, 735),
        1.0,
        1.0,
    )

    assert count == 86


def test_shortcut_count_rejects_first_spurious_digit_and_keeps_consensus_value():
    ocr = StatPanelOcr.__new__(StatPanelOcr)
    readings = iter([
        ("6", []),       # key-label/partial read from the first view
        ("2676", []),
        ("2676", []),
        ("2676", []),
    ])
    ocr._read_once = lambda _image: next(readings)

    from PIL import Image
    assert ocr.read_slot_count(Image.new("RGB", (36, 27))) == 2676


def test_shortcut_count_returns_none_without_independent_agreement():
    ocr = StatPanelOcr.__new__(StatPanelOcr)
    readings = iter([
        ("2676", []),
        ("1875", []),
        ("676", []),
        ("875", []),
    ])
    ocr._read_once = lambda _image: next(readings)

    from PIL import Image
    assert ocr.read_slot_count(Image.new("RGB", (36, 27))) is None


def test_shortcut_fast_cell_read_does_not_overwrite_complete_full_bar_value():
    ocr = StatPanelOcr.__new__(StatPanelOcr)
    ocr._read_shortcut_slot_counts = lambda *_args: {"1": 118}
    ocr.read_lines = lambda _image: [
        OcrLine("1180", y=54, x=38.5, left=24, right=53),
    ]

    from PIL import Image
    counts = ocr.read_shortcut_counts(Image.new("RGB", (297, 166)), {"1"})

    assert counts["1"] == 1180


def test_shortcut_full_validation_is_cached_when_fast_value_is_stable():
    ocr = StatPanelOcr.__new__(StatPanelOcr)
    ocr._read_shortcut_slot_counts = lambda *_args: {"1": 1180}
    calls = {"full": 0}

    def read_lines(_image):
        calls["full"] += 1
        return [OcrLine("1180", y=54, x=38.5, left=24, right=53)]

    ocr.read_lines = read_lines

    from PIL import Image
    image = Image.new("RGB", (297, 166))
    assert ocr.read_shortcut_counts(image, {"1"})["1"] == 1180
    assert ocr.read_shortcut_counts(image, {"1"})["1"] == 1180
    assert calls["full"] == 1


def test_shortcut_fast_read_uses_recovery_views_only_after_disagreement():
    ocr = StatPanelOcr.__new__(StatPanelOcr)
    readings = iter([
        ("2893", []),  # raw view has a thin outlined digit ambiguity
        ("2833", []),  # enlarged view
        ("2833", []),  # lower-right recovery
        ("2833", []),  # contrast recovery
    ])
    ocr._read_once = lambda _image: next(readings)

    from PIL import Image
    assert ocr.read_slot_count(Image.new("RGB", (38, 29)), allow_singleton=True, fast=True) == 2833


def test_shortcut_positioned_value_wins_over_neighbour_cell_merge():
    ocr = StatPanelOcr.__new__(StatPanelOcr)
    ocr._read_shortcut_slot_counts = lambda *_args: {"6": 26765}
    ocr.read_lines = lambda _image: [
        OcrLine("2676", y=122, x=114.3, left=90, right=140),
    ]

    from PIL import Image
    counts = ocr.read_shortcut_counts(Image.new("RGB", (297, 166)), {"6"})

    assert counts["6"] == 2676


def test_shortcut_numeric_model_wins_over_plausible_general_ocr_value():
    """A numeric cell read must not be overwritten by a whole-cell text read."""
    ocr = StatPanelOcr.__new__(StatPanelOcr)
    ocr._numeric_engine = object()
    ocr._read_numeric_field = lambda _image: "1487"
    ocr._read_once = lambda _image: ("1467", [])

    from PIL import Image
    assert ocr._read_shortcut_once(Image.new("RGB", (38, 20)))[0] == "1487"


def test_shortcut_numeric_views_prefer_clean_threshold_consensus_and_reject_conflict():
    assert ocr_module._select_shortcut_numeric_views(
        [(1209, "white170"), (1209, "white180"), (269, "soft-rgb")],
        previous=None,
    ) == 1209
    # 320 is a plausible OCR integer, but it conflicts with the threshold
    # interpretation 3204. Do not let majority voting invent a quantity.
    assert ocr_module._select_shortcut_numeric_views(
        [
            (320, "rgb"), (320, "gray"), (320, "r"),
            (3204, "white170"), (3204, "white180"),
        ],
        previous=None,
    ) is None


def test_shortcut_numeric_views_reject_1359_to_1959_substitution_but_keep_correction():
    # 3 -> 9 in the hundreds place is the reported same-length upward OCR
    # substitution.  A correction such as 40 -> 80 must remain possible so a
    # provisional false drop can be reversed by EconomyTracker.
    assert ocr_module._select_shortcut_numeric_views(
        [
            (1959, "rgb"),
            (1959, "gray"),
            (1959, "white170"),
            (1959, "white180"),
        ],
        previous=1359,
    ) is None
    assert ocr_module._select_shortcut_numeric_views(
        [
            (80, "rgb"),
            (80, "gray"),
            (80, "white170"),
            (80, "white180"),
        ],
        previous=40,
    ) == 80


def test_shortcut_fast_path_uses_latest_fast_cache_as_previous_value():
    # A full-bar cache for another slot must not hide the newest fast value for
    # this slot.  Without this regression guard, 1359 -> 1959 could be
    # returned as a fresh value instead of being held at 1359.
    from PIL import Image

    ocr = StatPanelOcr.__new__(StatPanelOcr)
    ocr._shortcut_last_full_counts = {"1": 2676}
    ocr._shortcut_last_fast_counts = {"7": 1359}
    ocr._shortcut_validation_signature = (("1", "7"), ())
    ocr._shortcut_last_cell_signatures = {}
    ocr._shortcut_last_cell_values = {}
    ocr._read_shortcut_slot_counts = lambda *_args, **_kwargs: {"7": 1959}

    expected = ocr.read_shortcut_counts(
        Image.new("RGB", (147, 77)),
        {"7"},
        allow_full_validation=False,
    )
    assert expected == {"7": 1359}
    # The rejected raw candidate must not erase the trusted cache and make the
    # identical bad value pass on the following fast frame.
    assert ocr.read_shortcut_counts(
        Image.new("RGB", (147, 77)),
        {"7"},
        allow_full_validation=False,
    ) == {"7": 1359}


def test_shortcut_full_validation_keeps_trusted_value_after_configuration_change():
    # The first full pass after Settings changes the enabled-cell signature
    # used to skip the stable-signature merge.  That allowed 1351 -> 1951 to
    # become the new displayed quantity even though 1351 was already trusted.
    from PIL import Image

    ocr = StatPanelOcr.__new__(StatPanelOcr)
    ocr._numeric_engine = object()
    ocr._shortcut_last_full_counts = {}
    ocr._shortcut_last_fast_counts = {"7": 1351}
    ocr._shortcut_last_validation_at = 1.0
    ocr._shortcut_validation_signature = (("1",), ())
    ocr._shortcut_last_cell_signatures = {}
    ocr._shortcut_last_cell_values = {}
    ocr._read_shortcut_slot_counts = lambda *_args, **_kwargs: {"7": 1951}
    ocr._shortcut_counts_from_records = lambda *_args, **_kwargs: {}
    ocr.read_lines = lambda _image: []

    assert ocr.read_shortcut_counts(
        Image.new("RGB", (147, 77)),
        {"7"},
        allow_full_validation=True,
    ) == {"7": 1351}


def test_shortcut_numeric_views_keep_aligned_prefixes_in_the_same_vote_families():
    """A dynamically cropped view must still count as threshold/colour OCR."""
    assert ocr_module._select_shortcut_numeric_views(
        [
            (1830, "aligned-white170"),
            (1830, "aligned-white180"),
            (1830, "aligned-rgb"),
        ],
        previous=None,
    ) == 1830


def test_shortcut_layout_pattern_tracks_every_narrow_one_without_fixed_slots():
    assert ocr_module._shortcut_layout_pattern(1105) == "11xx"
    assert ocr_module._shortcut_layout_pattern(1115) == "111x"
    assert ocr_module._shortcut_layout_pattern(1005) == "1xx1"
    assert ocr_module._shortcut_layout_pattern(2105) == "x1xx"
    assert ocr_module._shortcut_layout_pattern(2115) == "x11x"


def test_shortcut_layout_pattern_expands_left_as_narrow_ones_increase():
    from PIL import Image

    image = Image.new("RGB", (100, 20), (20, 20, 20))
    one_narrow = ocr_module._shortcut_pattern_quantity_crop(image, 2105)
    two_narrow = ocr_module._shortcut_pattern_quantity_crop(image, 2115)
    three_narrow = ocr_module._shortcut_pattern_quantity_crop(image, 1115)

    assert one_narrow is not None
    assert two_narrow is not None
    assert three_narrow is not None
    assert one_narrow.width < two_narrow.width < three_narrow.width


def test_right_aligned_quantity_crop_follows_the_glyph_run_width():
    """A leading 1 may be narrow, but both strings share the same right edge."""
    from PIL import Image, ImageDraw

    def make_quantity(glyph_ranges):
        image = Image.new("RGB", (40, 20), (20, 20, 20))
        draw = ImageDraw.Draw(image)
        # White blocks stand in for the connected vertical strokes of the
        # outlined game digits.  One coloured fragment on the left must not
        # become the quantity's left anchor.
        draw.rectangle((1, 4, 5, 13), fill=(255, 40, 40))
        for left, right in glyph_ranges:
            draw.rectangle((left, 4, right, 14), fill=(235, 235, 235))
        return image

    wide = make_quantity(((7, 11), (14, 18), (21, 25), (28, 33)))
    narrow_leading_one = make_quantity(((16, 17), (20, 24), (26, 29), (31, 33)))

    wide_crop = ocr_module._right_aligned_quantity_crop(wide)
    narrow_crop = ocr_module._right_aligned_quantity_crop(narrow_leading_one)

    assert wide_crop.width > narrow_crop.width
    assert wide_crop.width < wide.width
    assert narrow_crop.width < narrow_leading_one.width
    # The right-anchored glyph remains near the same right edge after the
    # variable left margin is removed.
    for crop in (wide_crop, narrow_crop):
        bright_columns = [
            x for x in range(crop.width)
            if any(crop.getpixel((x, y))[0] > 180 for y in range(crop.height))
        ]
        assert bright_columns
        assert bright_columns[-1] >= crop.width - 4


def test_shortcut_numeric_batch_passes_complete_bulk_observation_to_economy():
    class FakeNumeric:
        def __init__(self, values):
            self.values = values

        def read_fields(self, images):
            return {
                key: self.values.get(key.rsplit(":", 1)[-1], "")
                for key in images
                if self.values.get(key.rsplit(":", 1)[-1], "")
            }

    from PIL import Image
    image = Image.new("RGB", (38, 20))
    ocr = StatPanelOcr.__new__(StatPanelOcr)
    ocr._numeric_engine = FakeNumeric(
        {"rgb": "183.0", "white170": "1830", "white180": "1830"}
    )
    assert ocr._read_shortcut_numeric_batch(
        {"7": ("7", image)},
        blue_slot_ids=set(),
        previous_counts={},
    ) == {"7": 1830}

    ocr._numeric_engine = FakeNumeric(
        {"rgb": "1630", "white170": "1630", "white180": "1630"}
    )
    assert ocr._read_shortcut_numeric_batch(
        {"7": ("7", image)},
        blue_slot_ids=set(),
        previous_counts={"7": 1830},
    ) == {"7": 1630}


def test_numeric_shortcut_cache_skips_onnx_when_the_quantity_strip_is_unchanged():
    class CountingNumeric:
        def __init__(self):
            self.calls = 0

        def read_fields(self, images):
            self.calls += 1
            return {key: "1830" for key in images}

    from PIL import Image
    image = Image.new("RGB", (38, 41))
    numeric = CountingNumeric()
    ocr = StatPanelOcr.__new__(StatPanelOcr)
    ocr._numeric_engine = numeric
    ocr._shortcut_last_cell_signatures = {}
    ocr._shortcut_last_cell_values = {}

    first = ocr._read_shortcut_slot_counts(
        image,
        {"7"},
        slot_images={"7": image},
        previous_counts={},
    )
    second = ocr._read_shortcut_slot_counts(
        image,
        {"7"},
        slot_images={"7": image},
        previous_counts={"7": 1830},
    )

    assert first == {"7": 1830}
    assert second == {"7": 1830}
    assert numeric.calls == 1


def test_numeric_shortcut_cache_retries_an_unchanged_crop_after_blank_ocr():
    class BlankThenNumeric:
        def __init__(self):
            self.calls = 0

        def read_fields(self, images):
            self.calls += 1
            if self.calls == 1:
                return {}
            return {key: "1830" for key in images}

    from PIL import Image
    image = Image.new("RGB", (38, 41))
    numeric = BlankThenNumeric()
    ocr = StatPanelOcr.__new__(StatPanelOcr)
    ocr._numeric_engine = numeric
    ocr._shortcut_last_cell_signatures = {}
    ocr._shortcut_last_cell_values = {}

    assert ocr._read_shortcut_slot_counts(
        image,
        {"7"},
        slot_images={"7": image},
        previous_counts={},
    ) == {}
    assert ocr._read_shortcut_slot_counts(
        image,
        {"7"},
        slot_images={"7": image},
        previous_counts={},
    ) == {"7": 1830}
    assert numeric.calls == 2


def test_reset_shortcut_cache_forgets_the_previous_quantity_baseline():
    ocr = StatPanelOcr.__new__(StatPanelOcr)
    ocr._shortcut_last_cell_signatures = {"7:True": (1, 2)}
    ocr._shortcut_last_cell_values = {"7:True": 1830}
    ocr._shortcut_last_fast_counts = {"7": 1830}
    ocr._shortcut_last_full_counts = {"7": 1830}
    ocr._shortcut_last_validation_at = 12.0
    ocr._shortcut_validation_signature = (("7",), ("7",))

    ocr.reset_shortcut_cache()

    assert ocr._shortcut_last_cell_signatures == {}
    assert ocr._shortcut_last_cell_values == {}
    assert ocr._shortcut_last_fast_counts == {}
    assert ocr._shortcut_last_full_counts == {}
    assert ocr._shortcut_last_validation_at == 0.0
    assert ocr._shortcut_validation_signature is None


def test_numeric_engine_blank_does_not_fall_back_to_whole_cell_quantity():
    from PIL import Image
    image = Image.new("RGB", (38, 20))
    ocr = StatPanelOcr.__new__(StatPanelOcr)
    ocr._numeric_engine = object()
    ocr._read_shortcut_numeric_batch = lambda *args, **kwargs: {}
    ocr._read_numeric_field = lambda _image: ""
    ocr._read_once = lambda _image: ("320", [])
    ocr._shortcut_last_cell_signatures = {}
    ocr._shortcut_last_cell_values = {}

    assert ocr._read_shortcut_slot_counts(
        image,
        {"8"},
        {"8"},
        slot_images={"8": image},
        previous_counts={"8": 520},
    ) == {}


def test_explicitly_empty_shortcut_configuration_performs_no_full_bar_ocr():
    """An empty Settings selection must not scan all eight shortcut cells."""
    from PIL import Image

    ocr = StatPanelOcr.__new__(StatPanelOcr)
    calls = {"lines": 0}

    def read_lines(_image):
        calls["lines"] += 1
        return [OcrLine("1830", y=20, x=20, left=10, right=30)]

    ocr.read_lines = read_lines
    assert ocr.read_shortcut_counts(Image.new("RGB", (147, 77)), set()) == {}
    assert calls["lines"] == 0
