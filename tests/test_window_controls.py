"""Regression tests for the custom borderless window controls."""
from __future__ import annotations

from maple_analyzer.overlay import OverlayApp


class _StubRoot:
    def __init__(self) -> None:
        self._state = "normal"
        self._geometry = "760x900+40+40"
        self.calls: list[tuple[str, object]] = []

    def state(self) -> str:
        return self._state

    def geometry(self, value: str | None = None) -> str:
        if value is not None:
            self._geometry = value
        return self._geometry

    def overrideredirect(self, value: bool) -> None:
        self.calls.append(("overrideredirect", value))

    def update_idletasks(self) -> None:
        self.calls.append(("update_idletasks", None))

    def iconify(self) -> None:
        self.calls.append(("iconify", None))
        self._state = "iconic"


class _StubApp:
    def __init__(self) -> None:
        self.root = _StubRoot()
        self._normal_geometry = "760x900+40+40"
        self._borderless_restore_scheduled = False


def test_borderless_minimize_uses_tk_iconify_before_native_fallback():
    app = _StubApp()

    OverlayApp._minimize_window(app)

    assert app.root.calls == [
        ("overrideredirect", False),
        ("update_idletasks", None),
        ("iconify", None),
    ]
    assert app._normal_geometry == "760x900+40+40"
    assert app._minimize_in_progress is False
