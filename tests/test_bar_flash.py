from __future__ import annotations

from PIL import Image, ImageDraw

from maple_analyzer.bar_flash import BarFlashDetector


def _bar(fill_ratio: float, *, flash: bool = False) -> Image.Image:
    width, height = 100, 16
    image = Image.new("RGB", (width, height), (18, 24, 34))
    draw = ImageDraw.Draw(image)
    border = (255, 255, 255) if flash else (112, 124, 138)
    draw.rectangle((0, 0, width - 1, height - 1), outline=border, width=2)
    draw.rectangle((3, 3, width - 4, height - 4), fill=(44, 50, 62))
    fill_right = 3 + max(1, round((width - 7) * fill_ratio))
    draw.rectangle((3, 3, fill_right, height - 4), fill=(212, 40, 54))
    if flash:
        # The in-game effect is a short bright frame/highlight, not a change
        # to the amount-filled portion of the bar.
        draw.line((2, 2, width - 3, 2), fill=(255, 255, 255), width=1)
        draw.line((2, height - 3, width - 3, height - 3), fill=(255, 255, 255), width=1)
    return image


def test_fill_width_changes_do_not_trigger_a_flash():
    detector = BarFlashDetector()
    base = _bar(0.25)
    for _ in range(4):
        assert detector.update({"hp": base}) == ()

    # Damage/heal changes the red fill width but leaves the bar frame alone.
    assert detector.update({"hp": _bar(0.75)}) == ()
    assert detector.update({"hp": _bar(0.10)}) == ()


def test_fill_changes_above_a_shifted_bottom_frame_do_not_trigger():
    """A shorter client can place the horizontal frame below the crop top."""
    def shifted(fill_ratio: float) -> Image.Image:
        image = Image.new("RGB", (100, 14), (18, 24, 34))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 99, 9), fill=(44, 50, 62))
        fill_right = 3 + max(1, round(94 * fill_ratio))
        draw.rectangle((3, 0, fill_right, 8), fill=(212, 40, 54))
        draw.line((3, 9, 96, 9), fill=(255, 255, 255), width=1)
        return image

    detector = BarFlashDetector()
    for _ in range(4):
        assert detector.update({"hp": shifted(0.25)}) == ()
    assert detector.update({"hp": shifted(0.75)}) == ()
    assert detector.update({"hp": shifted(0.10)}) == ()


def test_bright_bar_frame_emits_one_edge_triggered_event():
    detector = BarFlashDetector()
    base = _bar(0.25)
    flash = _bar(0.60, flash=True)
    for _ in range(4):
        detector.update({"hp": base})

    assert detector.update({"hp": flash}) == ("hp",)
    # A flash can remain visible for more than one 0.3s sample; it is still
    # only one drinking event until the frame returns to normal.
    assert detector.update({"hp": flash}) == ()
    assert detector.update({"hp": base}) == ()
    assert detector.update({"hp": flash}) == ("hp",)


def test_hp_and_mp_flashes_are_independent():
    detector = BarFlashDetector()
    base = {"hp": _bar(0.25), "mp": _bar(0.25)}
    flash_hp = {"hp": _bar(0.60, flash=True), "mp": base["mp"]}
    flash_mp = {"hp": base["hp"], "mp": _bar(0.60, flash=True)}
    for _ in range(4):
        assert detector.update(base) == ()

    assert detector.update(flash_hp) == ("hp",)
    assert detector.update(base) == ()
    assert detector.update(flash_mp) == ("mp",)
