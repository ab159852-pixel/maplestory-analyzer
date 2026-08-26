"""Regression tests for the economy OCR crops derived from the UI sample."""
from __future__ import annotations

from conftest import SAMPLE_IMAGE
from maple_analyzer.capture import StaticImageCapture
from maple_analyzer.regions import (
    AUXILIARY_BOXES,
    PICKUP_LINE_BOXES,
    SHORTCUT_SLOT_BOXES,
)


def test_static_auxiliary_capture_returns_full_and_fast_subcrops():
    regions = StaticImageCapture(SAMPLE_IMAGE).grab_auxiliary()

    assert set(regions) >= {"pickup", "shortcut"}
    assert set(f"pickup:{line}" for line in PICKUP_LINE_BOXES) <= regions.keys()
    assert set(f"shortcut:{slot}" for slot in SHORTCUT_SLOT_BOXES) <= regions.keys()
    assert regions["pickup"].size == (
        AUXILIARY_BOXES["pickup"][2] - AUXILIARY_BOXES["pickup"][0],
        AUXILIARY_BOXES["pickup"][3] - AUXILIARY_BOXES["pickup"][1],
    )
    for slot in SHORTCUT_SLOT_BOXES:
        image = regions[f"shortcut:{slot}"]
        assert image.width > 0 and image.height > 0
    # The eight OCR crops must tile the measured parent frame.  This catches
    # both the old oversized parent and the old inner boxes that cut a digit.
    top_row = [regions[f"shortcut:{slot}"] for slot in ("1", "2", "3", "4")]
    bottom_row = [regions[f"shortcut:{slot}"] for slot in ("5", "6", "7", "8")]
    assert sum(image.width for image in top_row) == regions["shortcut"].width
    assert sum(image.width for image in bottom_row) == regions["shortcut"].width
    assert top_row[0].height + bottom_row[0].height == regions["shortcut"].height
    for line in PICKUP_LINE_BOXES:
        image = regions[f"pickup:{line}"]
        assert image.width > 0 and image.height > 0
