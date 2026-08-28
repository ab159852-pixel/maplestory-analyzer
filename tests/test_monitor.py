"""Regression tests for the settings-driven background OCR selection."""
from __future__ import annotations

from PIL import Image

from maple_analyzer.monitor import BackgroundMonitor
from maple_analyzer.settings import PotionSlotConfig


def _regions():
    return {
        "shortcut": Image.new("RGB", (147, 77)),
        **{
            f"shortcut:{slot}": Image.new("RGB", (30, 30))
            for slot in range(1, 9)
        },
    }


class _RecordingOcr:
    def __init__(self):
        self.calls = []

    def read_shortcut_counts(self, image, required, blue, **kwargs):
        del image
        self.calls.append((
            set(required),
            set(blue),
            set((kwargs.get("slot_images") or {}).keys()),
            kwargs.get("live"),
        ))
        return {"1": 2676, "8": 9999}


def test_empty_potion_settings_do_not_start_shortcut_ocr():
    ocr = _RecordingOcr()
    monitor = BackgroundMonitor.__new__(BackgroundMonitor)
    monitor.ocr = ocr

    assert monitor._read_potion_counts(_regions(), tuple(), tuple()) == {}
    assert ocr.calls == []


def test_background_ocr_receives_only_enabled_shortcut_cell():
    ocr = _RecordingOcr()
    monitor = BackgroundMonitor.__new__(BackgroundMonitor)
    monitor.ocr = ocr
    configured = (PotionSlotConfig(slot="1", kind="hp", cost=10, enabled=True),)

    assert monitor._read_potion_counts(_regions(), configured, configured) == {"1": 2676}
    assert ocr.calls == [({"1"}, set(), {"1"}, True)]


def test_background_ocr_receives_all_enabled_shortcut_cells_in_settings_order():
    class _TwoCellOcr(_RecordingOcr):
        def read_shortcut_counts(self, image, required, blue, **kwargs):
            super().read_shortcut_counts(image, required, blue, **kwargs)
            return {"6": 2676, "7": 1875}

    ocr = _TwoCellOcr()
    monitor = BackgroundMonitor.__new__(BackgroundMonitor)
    monitor.ocr = ocr
    configured = (
        PotionSlotConfig(slot="6", kind="hp", cost=10, enabled=True),
        PotionSlotConfig(slot="7", kind="mp", cost=20, enabled=True),
    )

    assert monitor._read_potion_counts(_regions(), configured, configured) == {
        "6": 2676,
        "7": 1875,
    }
    assert ocr.calls == [({"6", "7"}, {"7"}, {"6", "7"}, True)]


def test_pickup_row_retries_after_an_empty_cached_ocr_result():
    class _PickupOcr:
        def __init__(self):
            self.row_results = iter(("", "獲取楓幣。(+144)"))
            self.detector_calls = 0

        def read_text_field(self, _image):
            return next(self.row_results)

        def read_lines(self, _image):
            self.detector_calls += 1
            return []

    pickup = Image.new("RGB", (271, 195), "white")
    regions = {
        "pickup": pickup,
        "pickup_wide": Image.new("RGB", (591, 340), "black"),
        "pickup:0": Image.new("RGB", (271, 16), "white"),
    }
    ocr = _PickupOcr()
    monitor = BackgroundMonitor.__new__(BackgroundMonitor)
    monitor.ocr = ocr
    monitor._pickup_feed_signature = None
    monitor._pickup_detection_signature = None
    monitor._pickup_detected_lines = []
    monitor._pickup_line_signatures = {}
    monitor._pickup_line_values = {}
    monitor._next_pickup_detection = 0.0

    # The first recognition miss causes the detector to be attempted and its
    # empty result cached. A later pass over the unchanged feed must still retry
    # the row instead of returning that stale empty result forever.
    assert monitor._read_pickup_lines(regions, now=1.0) == [("", 16.0)]
    assert monitor._read_pickup_lines(regions, now=1.1) == [("獲取楓幣。(+144)", 16.0)]
    # The first miss searches both configured detector crops; the cached empty
    # result prevents any additional detector call on the second pass.
    assert ocr.detector_calls == 2
