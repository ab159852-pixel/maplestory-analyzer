from __future__ import annotations

import json
from pathlib import Path


def test_app_owned_customtkinter_theme_contains_required_widget_sections():
    theme_path = Path(__file__).resolve().parents[1] / "assets" / "maple_insight_dark_blue.json"

    theme = json.loads(theme_path.read_text(encoding="utf-8"))

    assert theme["CTk"]["fg_color"]
    assert theme["CTkButton"]["fg_color"]
    assert theme["CTkFrame"]["fg_color"]
    assert theme["CTkFont"]["Windows"]["family"] == "Roboto"
