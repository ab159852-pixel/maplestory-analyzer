"""Do the crops survive a different resolution and aspect ratio?

The static fixture is a complete reference-window image, with boxes measured
at REFERENCE_CLIENT_SIZE (the legacy name for 1351x800). Live HWND capture
uses the separate window-to-client mapping tests in test_regions.py. These
tests keep the static OCR contract intact and measure the existing transform
against a real second screenshot rather than a resized copy of the first one
(resizing a screenshot does not reproduce how the game re-renders its UI at a
different size, so it proves nothing).
"""
from __future__ import annotations

import pytest
from PIL import Image

from conftest import SAMPLE_IMAGE_1920
from maple_analyzer.ocr import StatPanelOcr
from maple_analyzer.parser import parse_fields
from maple_analyzer.regions import (
    FIELD_BOXES,
    REFERENCE_CLIENT_SIZE,
    STAT_PANEL_BOX,
    scale_box,
)

# Ground truth read off samples/maple_story_ui_1920.jpg by eye.
TRUTH_1920 = {"level": 44, "hp": (824, 824), "mp": (2816, 2816), "exp_pct": 83.31}

CLIENT_SIZES = [(1351, 800), (1366, 768), (1920, 1077), (1280, 720), (2560, 1440)]


@pytest.fixture(scope="module")
def snapshot_1920():
    image = Image.open(SAMPLE_IMAGE_1920).convert("RGB")
    ocr = StatPanelOcr()
    text = {
        name: ocr.read_field(image.crop(scale_box(box, image.size).as_tuple()))
        for name, box in FIELD_BOXES.items()
    }
    return parse_fields(text), text


def test_reference_size_is_the_identity_case():
    for box in list(FIELD_BOXES.values()) + [STAT_PANEL_BOX]:
        assert scale_box(box, REFERENCE_CLIENT_SIZE).as_tuple() == box


@pytest.mark.parametrize("client", CLIENT_SIZES)
def test_field_boxes_stay_inside_the_panel_box(client):
    """Catches a scaling regression at any resolution without needing a
    screenshot for each one."""
    panel = scale_box(STAT_PANEL_BOX, client)
    for name, raw in FIELD_BOXES.items():
        box = scale_box(raw, client)
        assert panel.left <= box.left < box.right <= panel.right, name
        assert panel.top <= box.top < box.bottom <= panel.bottom, name
        assert box.right - box.left >= 40, f"{name} too narrow at {client}"
        assert box.bottom - box.top >= 10, f"{name} too short at {client}"


@pytest.mark.parametrize("client", CLIENT_SIZES)
def test_panel_box_stays_inside_the_client(client):
    panel = scale_box(STAT_PANEL_BOX, client)
    assert 0 <= panel.left < panel.right <= client[0]
    assert 0 <= panel.top < panel.bottom <= client[1]


def test_level_and_mp_read_correctly_at_1920(snapshot_1920):
    """The geometry claim: proportional crops still land on the text at a
    resolution and aspect ratio the boxes were never measured at."""
    snap, text = snapshot_1920
    assert snap.level == TRUTH_1920["level"], text
    assert (snap.mp_cur, snap.mp_max) == TRUTH_1920["mp"], text


def test_exp_percentage_reads_correctly_at_1920(snapshot_1920):
    snap, text = snapshot_1920
    assert snap.exp_pct == TRUTH_1920["exp_pct"], text


def test_hp_reads_correctly_at_1920(snapshot_1920):
    snap, _text = snapshot_1920
    assert (snap.hp_cur, snap.hp_max) == TRUTH_1920["hp"]


def test_hp_digits_are_stable_at_1920(snapshot_1920):
    """The digits remain correct even when OCR changes the separator shape."""
    _snap, text = snapshot_1920
    hp = text["HP"].replace(" ", "")
    assert "824" in hp
    assert hp.count("824") == 2, hp  # both numbers present, both correct
