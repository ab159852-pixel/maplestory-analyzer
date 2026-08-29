"""Occlusion detection: is another window covering the stat panel?

`mss` grabs a screen *region*, not the game's pixels, so anything resting on
that strip is what reaches OCR. Live capture (2026-08-17) caught the app
reading its own log back off a terminal covering the panel, which parsed as
MP 1/2 and booked the whole bar as loss.

The Win32 lookup is injected, so the logic here is testable off Windows like
the rest of the suite.
"""
from __future__ import annotations

import pytest

from maple_analyzer.capture import (
    GameWindowCapture,
    TARGET_WINDOW_CAPTURE_UNAVAILABLE,
    context_sample_points,
    field_sample_points,
    panel_is_obscured,
)
from maple_analyzer.regions import (
    CONTEXT_BOXES,
    FIELD_BOXES,
    REFERENCE_CLIENT_SIZE,
    scale_box,
)

GAME = 1000
OTHER = 2000

POINTS = field_sample_points(REFERENCE_CLIENT_SIZE)


def _always(hwnd):
    return lambda _x, _y: hwnd


def test_clear_panel_is_not_flagged():
    assert panel_is_obscured(POINTS, GAME, _always(GAME)) is False


def test_fully_covered_panel_is_flagged():
    assert panel_is_obscured(POINTS, GAME, _always(OTHER)) is True


@pytest.mark.parametrize("index", range(len(POINTS)))
def test_partially_covered_panel_is_flagged(index):
    """Partial coverage is the *dangerous* case, not a lesser one: a window
    clipping only some digits leaves the field's label readable, so the value
    still parses -- just wrong. Any covered sample point must flag."""
    covered = POINTS[index]

    def window_at(x, y):
        return OTHER if (x, y) == covered else GAME

    assert panel_is_obscured(POINTS, GAME, window_at) is True


def test_game_child_windows_do_not_count_as_covering():
    """window_at resolves to the *root* window, so the game's own child
    windows/controls read as the game. A false positive here would stop
    tracking entirely, which is worse than the bug being fixed."""
    assert panel_is_obscured(POINTS, GAME, _always(GAME)) is False


def test_sample_points_lie_inside_every_field_box():
    """Guards an off-by-one that would probe just outside the fields and so
    never detect anything -- a silently useless check."""
    hit = {name: 0 for name in FIELD_BOXES}
    for name, raw in FIELD_BOXES.items():
        box = scale_box(raw, REFERENCE_CLIENT_SIZE)
        for x, y in POINTS:
            if box.left <= x < box.right and box.top <= y < box.bottom:
                hit[name] += 1
    assert all(count > 0 for count in hit.values()), hit
    # every point must belong to some field box
    assert sum(hit.values()) >= len(POINTS)


def test_sample_points_scale_with_the_client():
    small = field_sample_points((1351, 800))
    large = field_sample_points((1920, 1077))
    assert len(small) == len(large)
    assert max(x for x, _ in large) > max(x for x, _ in small)


def test_context_sample_points_cover_map_and_job_at_2k_geometry():
    points = context_sample_points(
        (2560, 1431),
        window_size=(2560, 1467),
        client_offset=(0, 36),
    )
    assert len(points) == 8
    for name in ("map", "job"):
        if name == "map":
            from maple_analyzer.regions import scale_window_top_left_box_to_client

            box = scale_window_top_left_box_to_client(
                CONTEXT_BOXES[name], (2560, 1431), (2560, 1467), (0, 36)
            )
        else:
            from maple_analyzer.regions import scale_window_box_to_client

            box = scale_window_box_to_client(
                CONTEXT_BOXES[name], (2560, 1431), (2560, 1467), (0, 36)
            )
        name_points = points[:4] if name == "map" else points[4:]
        assert all(box.left <= x < box.right and box.top <= y < box.bottom for x, y in name_points)


def test_context_never_falls_back_to_foreground_desktop_pixels():
    capture = GameWindowCapture.__new__(GameWindowCapture)
    capture._client_rect_on_screen = lambda: (0, 0, 2560, 1440)
    capture._remember_capture_geometry = lambda *_args, **_kwargs: None
    capture._try_window_frame = lambda _rect: None
    capture._target_window_capture_error = (
        lambda: f"{TARGET_WINDOW_CAPTURE_UNAVAILABLE}; WGC test failure"
    )

    class DesktopMustNotBeRead:
        def grab(self, _box):
            raise AssertionError("context OCR must not read foreground desktop pixels")

    capture._mss = DesktopMustNotBeRead()
    with pytest.raises(RuntimeError, match=TARGET_WINDOW_CAPTURE_UNAVAILABLE):
        capture._grab_context()


def test_obscured_message_is_localised():
    """The three capture states are routine, expected conditions users hit
    constantly, so they get real translations rather than leaking raw English
    into a zh UI -- same treatment as minimized/not-found."""
    from maple_analyzer.capture import PANEL_OBSCURED
    from maple_analyzer.overlay import OverlayApp
    from maple_analyzer.settings import Settings

    class Stub:
        _settings = Settings()  # zh by default
        _t = OverlayApp._t
        _localize_error = OverlayApp._localize_error

    stub = Stub()
    localized = stub._localize_error(PANEL_OBSCURED)
    assert localized != PANEL_OBSCURED
    assert "被" in localized  # zh string, not raw English

    stub._settings = Settings(language="en")
    assert stub._localize_error(PANEL_OBSCURED) == (
        "Stat panel is covered; live capture is unavailable"
    )

    assert stub._localize_error(TARGET_WINDOW_CAPTURE_UNAVAILABLE) == (
        "Target-window capture unavailable; retrying WGC"
    )
