"""Regression tests for the settings-driven background OCR selection."""
from __future__ import annotations

import threading
import queue

from PIL import Image, ImageDraw

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


def test_monitor_stop_releases_native_capture_source_with_one_total_budget():
    class Source:
        def __init__(self):
            self.closed = threading.Event()

        def close(self):
            self.closed.set()

    monitor = BackgroundMonitor.__new__(BackgroundMonitor)
    monitor.source = Source()
    monitor._stop = threading.Event()
    monitor._status_enabled = threading.Event()
    monitor._aux_enabled = threading.Event()
    monitor._potion_request = threading.Event()
    monitor._pickup_request = threading.Event()
    monitor._context_request = threading.Event()
    monitor._potion_scan_active = threading.Event()
    monitor._threads = []

    monitor.stop(total_timeout=0.2)

    assert monitor.source.closed.is_set()
    assert monitor._stop.is_set()


def test_pickup_capture_queue_keeps_latest_stack_without_blocking():
    monitor = BackgroundMonitor.__new__(BackgroundMonitor)
    monitor._pickup_frame_queue = queue.Queue(maxsize=2)

    monitor._queue_pickup_frame(1.0, {"pickup": "first"})
    monitor._queue_pickup_frame(2.0, {"pickup": "second"})
    monitor._queue_pickup_frame(3.0, {"pickup": "latest"})

    assert monitor._pickup_frame_queue.get_nowait() == (
        2.0,
        {"pickup": "second"},
    )
    assert monitor._pickup_frame_queue.get_nowait() == (
        3.0,
        {"pickup": "latest"},
    )


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


def _pickup_feed(*rows):
    """Build a black toast surface with scalable text-like horizontal rows."""
    image = Image.new("RGB", (286, 195), "black")
    draw = ImageDraw.Draw(image)
    for y, colour, width in rows:
        # A few black cuts make the visual fingerprint resemble separate
        # glyphs while retaining deterministic synthetic pixels.
        draw.rectangle((80, y, 80 + width, y + 10), fill=colour)
        for x in range(88, 80 + width, 13):
            draw.rectangle((x, y, x + 2, y + 10), fill="black")
    return image


def test_dynamic_pickup_segmentation_skips_six_yellow_rows_and_reads_money_only():
    class _PickupOcr:
        def __init__(self):
            self.calls = 0

        def read_text_field(self, _image):
            self.calls += 1
            return "獲取楓幣。(+275)"

        def read_lines(self, _image):
            raise AssertionError("strong row OCR must not invoke detector fallback")

    rows = [
        (20 + index * 14, "yellow", 120 - index * 4)
        for index in range(6)
    ] + [(104, "white", 92)]
    ocr = _PickupOcr()
    monitor = BackgroundMonitor(None, ocr)

    assert monitor._read_pickup_lines({"pickup": _pickup_feed(*rows)}, now=1.0) == [
        ("獲取楓幣。(+275)", 109.0),
    ]
    assert ocr.calls == 1
    assert monitor._pickup_scan_confident is True


def test_scrolled_pickup_row_reuses_visual_ocr_and_reads_only_new_message():
    class _PickupOcr:
        def __init__(self):
            self.values = iter(("獲取楓幣。(+100)", "獲取楓幣。(+200)"))
            self.calls = 0

        def read_text_field(self, _image):
            self.calls += 1
            return next(self.values)

        def read_lines(self, _image):
            raise AssertionError("strong row OCR must not invoke detector fallback")

    ocr = _PickupOcr()
    monitor = BackgroundMonitor(None, ocr)
    first = _pickup_feed((100, "white", 86))
    # The +100 row moves upward unchanged and +200 is the only new visual row.
    second = _pickup_feed(
        (86, "white", 86),
        (100, "white", 72),
    )

    assert monitor._read_pickup_lines({"pickup": first}, now=1.0) == [
        ("獲取楓幣。(+100)", 105.0),
    ]
    assert monitor._read_pickup_lines({"pickup": second}, now=1.1) == [
        ("獲取楓幣。(+100)", 91.0),
        ("獲取楓幣。(+200)", 105.0),
    ]
    assert ocr.calls == 2


def test_uncertain_white_pickup_row_does_not_publish_a_false_empty_boundary():
    class _PickupOcr:
        def read_text_field(self, _image):
            return ""

        def read_lines(self, _image):
            return []

    monitor = BackgroundMonitor(None, _PickupOcr())

    assert monitor._read_pickup_lines(
        {"pickup": _pickup_feed((100, "white", 86))},
        now=1.0,
    ) == []
    assert monitor._pickup_scan_confident is False


def test_partial_money_stack_cannot_advance_the_duplicate_tracking_boundary():
    class _PickupOcr:
        def __init__(self):
            self.values = iter(("獲取楓幣。(+100)", ""))

        def read_text_field(self, _image):
            return next(self.values)

        def read_lines(self, _image):
            return []

    first = Image.new("RGB", (100, 18), "black")
    second = Image.new("RGB", (100, 18), "black")
    ImageDraw.Draw(first).rectangle((20, 3, 70, 13), fill="white")
    ImageDraw.Draw(second).rectangle((20, 3, 82, 13), fill="white")
    monitor = BackgroundMonitor(None, _PickupOcr())

    assert monitor._read_dynamic_pickup_rows([
        (first, 90.0),
        (second, 104.0),
    ]) == [("獲取楓幣。(+100)", 90.0)]
    # The readable +100 row must not make the partial result look complete.
    # Otherwise the missed second row is forgotten and counted again as new
    # when it becomes readable on the next 0.1-0.2s frame.
    assert monitor._pickup_scan_confident is False
