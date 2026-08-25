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
