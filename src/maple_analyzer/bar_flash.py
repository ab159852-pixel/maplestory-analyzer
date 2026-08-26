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

    Only the perimeter and a narrow centre strip are used.  The perimeter is
    stable when the amount changes, while the centre strip captures the pale
    highlight that accompanies the flash.  Values are normalized so the same
    logic works at every captured resolution.
    """
    if not isinstance(image, Image.Image) or image.width < 8 or image.height < 6:
        return None
    rgb = image.convert("RGB")
    width, height = rgb.size
    pixels = []
    for y in range(height):
        for x in range(width):
            if x < 3 or x >= width - 3 or y < 3 or y >= height - 3:
                pixels.append(rgb.getpixel((x, y)))
    # The middle row avoids most text above the bar and the empty track still
    # contributes only through the fixed frame/highlight colours.
    center_y = height // 2
    pixels.extend(rgb.getpixel((x, center_y)) for x in range(3, width - 3))
    if not pixels:
        return None

    red = green = blue = luminance = saturation = 0.0
    bright = pale = red_like = blue_like = 0
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
        red_like / total,
        blue_like / total,
    )


def _distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    # Colour channels and the pale/highlight ratios carry most of the signal;
    # the raw luminance/saturation terms provide a useful fallback for clients
    # that render the flash as white rather than pink/blue.
    weights = (1.0, 1.0, 1.0, 0.8, 0.8, 1.4, 1.6, 1.2, 1.2)
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
