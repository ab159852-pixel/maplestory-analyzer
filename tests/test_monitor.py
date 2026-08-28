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
