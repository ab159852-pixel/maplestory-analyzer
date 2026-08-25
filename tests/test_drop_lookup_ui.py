from __future__ import annotations

from maple_analyzer.overlay import OverlayApp


def test_opening_drop_lookup_card_requests_detected_map_once():
    app = object.__new__(OverlayApp)
    app._drop_lookup_expanded = False
    app._drop_lookup_requested_map = "第3軍營"
    app._drop_lookup_summary = None
    app._drop_lookup_loading_map = None
    app._current_map_name = lambda: "第3軍營"
    applied: list[bool] = []
    requested: list[bool] = []
    app._apply_drop_lookup_expanded = lambda: applied.append(app._drop_lookup_expanded)
    app._on_drop_lookup_clicked = lambda: requested.append(True)

    OverlayApp._toggle_drop_lookup_card(app)

    assert applied == [True]
    assert requested == [True]


def test_opening_drop_lookup_card_without_map_does_not_start_request():
    app = object.__new__(OverlayApp)
    app._drop_lookup_expanded = False
    app._drop_lookup_requested_map = None
    app._drop_lookup_summary = None
    app._drop_lookup_loading_map = None
    app._current_map_name = lambda: None
    requested: list[bool] = []
    app._apply_drop_lookup_expanded = lambda: None
    app._on_drop_lookup_clicked = lambda: requested.append(True)

    OverlayApp._toggle_drop_lookup_card(app)

    assert app._drop_lookup_expanded is True
    assert requested == []


def test_detected_map_after_panel_open_starts_lookup():
    class Label:
        def configure(self, **_kwargs):
            return None

    app = object.__new__(OverlayApp)
    app._drop_lookup_map_label = Label()
    app._drop_lookup_expanded = True
    app._drop_lookup_requested_map = None
    app._drop_lookup_cache = {}
    app._drop_lookup_summary = None
    app._drop_lookup_loading_map = None
    app._drop_lookup_error = None
    app._drop_detail_expanded = set()
    app._current_map_name = lambda: "第3軍營"
    app._t = lambda key, **_kwargs: key
    app._render_drop_lookup = lambda: None
    requested: list[bool] = []
    app._on_drop_lookup_clicked = lambda: requested.append(True)

    OverlayApp._render_drop_lookup_header(app)

    assert app._drop_lookup_requested_map == "第3軍營"
    assert requested == [True]
