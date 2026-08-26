"""Detect the short visual flash drawn around MapleStory's HP/MP bars.

The effect is deliberately treated as an edge-triggered signal.  A normal
HP/MP value change alters the filled width of a bar, while the drink effect
temporarily changes the bar frame/highlight.  Keeping a small per-resource
baseline lets the detector ignore the former and emit at most one event for
the latter, even when the same flash is visible for several captured frames.
"""
from __future__ import annotations

import colorsys
from dataclasses import dataclass, field
from typing import Any

from PIL import Image


# The detector is intentionally conservative.  It must not turn ordinary
# damage animation, redraws, or a one-frame capture artefact into a potion
# event.  Hysteresis also prevents a flash lasting 2-3 frames from repeating.
FLASH_TRIGGER_SCORE = 0.18
FLASH_CLEAR_SCORE = 0.10
BASELINE_BLEND = 0.04
WARMUP_SAMPLES = 4


def _signature(image: Any) -> tuple[float, ...] | None:
    """Return a compact colour signature for the fixed bar frame.

    Only the outermost perimeter is used.  The fill itself changes on every
    damage/heal tick, including the first interior row in the game's small
    crop, so a wider edge band would turn ordinary HP/MP changes into flashes.
    Values are normalized so the same logic works at every captured
    resolution.
    """
    if not isinstance(image, Image.Image) or image.width < 8 or image.height < 6:
        return None
    rgb = image.convert("RGB")
    width, height = rgb.size
    # Do not sample the middle of the bar: its fill width changes on every
    # damage/heal tick. The crop can land above the top frame on a shorter
    # client, so find the brightest low-saturation horizontal edge row instead
    # of assuming it is always the first row. The narrow side strips provide a
    # second edge signal without seeing the horizontal fill.
    side = max(1, min(3, width // 40))
    pixels = []
    for y in range(height):
        pixels.extend(rgb.getpixel((x, y)) for x in range(side))
        pixels.extend(rgb.getpixel((x, y)) for x in range(max(side, width - side), width))
    interior_left = side
    interior_right = max(interior_left + 1, width - side)

    def edge_score(y: int) -> tuple[float, float, float]:
        row = [rgb.getpixel((x, y)) for x in range(interior_left, interior_right)]
        if not row:
            return 0.0, 0.0, 0.0
        pale = neutral_bright = luminance = 0.0
        for red, green, blue in row:
            _hue, saturation, value = colorsys.rgb_to_hsv(
                red / 255.0, green / 255.0, blue / 255.0
            )
            pale += saturation <= 0.40 and value >= 0.45
            neutral_bright += saturation <= 0.34 and value >= 0.72
            luminance += (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255.0
        total = float(len(row))
        # Prefer a row that is broadly pale/neutral, then one with the highest
        # luminance. This picks the actual white frame over the blue-gray
        # client edge that can be visible below a clipped status panel.
        return neutral_bright / total, pale / total, luminance / total

    edge_row = max(range(height), key=edge_score)
    neutral_score, pale_score, _luminance = edge_score(edge_row)
    if neutral_score >= 0.45 or pale_score >= 0.70:
        pixels.extend(rgb.getpixel((x, edge_row)) for x in range(width))
    if not pixels:
        return None

    red = green = blue = luminance = saturation = 0.0
    bright = pale = neutral_bright = red_like = blue_like = 0
    for r, g, b in pixels:
        red += r / 255.0
        green += g / 255.0
        blue += b / 255.0
        value = max(r, g, b) / 255.0
        _hue, sat, _value = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        luminance += (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
        saturation += sat
        bright += value >= 0.82
        pale += sat <= 0.34 and value >= 0.55
        neutral_bright += sat <= 0.28 and value >= 0.72
        red_like += r >= g * 1.35 and r >= b * 1.35 and r >= 75
        blue_like += b >= r * 1.18 and b >= g * 1.03 and b >= 75
    total = float(len(pixels))
    return (
        red / total,
        green / total,
        blue / total,
        luminance / total,
        saturation / total,
        bright / total,
        pale / total,
        neutral_bright / total,
        red_like / total,
        blue_like / total,
    )


def _distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    # Colour channels and the pale/highlight ratios carry most of the signal;
    # the raw luminance/saturation terms provide a useful fallback for clients
    # that render the flash as white rather than pink/blue.
    weights = (1.0, 1.0, 1.0, 0.8, 0.8, 1.4, 1.6, 2.0, 1.2, 1.2)
    return sum(weight * abs(a - b) for a, b, weight in zip(left, right, weights))


@dataclass
class BarFlashDetector:
    """Turn bar-frame changes into one-shot ``hp``/``mp`` events."""

    _baseline: dict[str, tuple[float, ...]] = field(default_factory=dict)
    _samples: dict[str, int] = field(default_factory=dict)
    _active: dict[str, bool] = field(default_factory=dict)

    def reset(self) -> None:
        self._baseline.clear()
        self._samples.clear()
        self._active.clear()

    def update(self, images: dict[str, Any]) -> tuple[str, ...]:
        events: list[str] = []
        for resource in ("hp", "mp"):
            signature = _signature(images.get(resource))
            if signature is None:
                continue
            baseline = self._baseline.get(resource)
            if baseline is None:
                self._baseline[resource] = signature
                self._samples[resource] = 1
                self._active[resource] = False
                continue
            samples = self._samples.get(resource, 0)
            if samples < WARMUP_SAMPLES:
                count = samples + 1
                self._baseline[resource] = tuple(
                    (old * samples + new) / count
                    for old, new in zip(baseline, signature)
                )
                self._samples[resource] = count
                continue

            score = _distance(baseline, signature)
            active = self._active.get(resource, False)
            if active:
                if score < FLASH_CLEAR_SCORE:
                    self._active[resource] = False
                    self._baseline[resource] = tuple(
                        old * (1.0 - BASELINE_BLEND) + new * BASELINE_BLEND
                        for old, new in zip(baseline, signature)
                    )
                # Do not update the baseline while the flash is active.
                continue
            if score >= FLASH_TRIGGER_SCORE:
                self._active[resource] = True
                events.append(resource)
                continue
            self._baseline[resource] = tuple(
                old * (1.0 - BASELINE_BLEND) + new * BASELINE_BLEND
                for old, new in zip(baseline, signature)
            )
        return tuple(events)
