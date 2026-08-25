"""Crop-box definitions for the stat panel and its fields.

The boxes are measured in a reference game viewport, then mapped to the actual
client pixels reported by the capture backend on every frame.  The mapping keeps
the game's aspect ratio, centers horizontal letterboxing, and anchors the bottom
HUD to the bottom of the client.  That is more stable than independently
stretching x/y coordinates when a user resizes a window or moves it between DPI
scales.
"""
from __future__ import annotations

from dataclasses import dataclass

REFERENCE_CLIENT_SIZE = (1351, 800)  # size of samples/maple_story_ui.jpg

# Whole stat panel, in absolute pixels at REFERENCE_CLIENT_SIZE. Grabbed once per
# tick with a single mss.grab() call; FIELD_BOXES below are sliced out of it
# in-memory (no extra screen captures).
STAT_PANEL_BOX = (260, 758, 900, 800)  # (left, top, right, bottom)

# Per-field boxes, same reference frame as STAT_PANEL_BOX, generously padded
# around each field's label+value text (measured off samples/maple_story_ui.jpg,
# see commit history for the crop-and-inspect process). Recognition-only OCR
# (no detection) runs on each of these individually -- see ocr.py's read_field()
# docstring for why this replaced running detection over the whole panel
# (detection was ~600ms/call, the actual OCR bottleneck; recognition-only on a
# small pre-cropped box is ~15ms).
FIELD_BOXES = {
    # Keep this crop focused on the LV/value glyphs.  Widening it into the
    # adjacent job label makes RapidOCR choose Chinese job text instead of the
    # orange level digits, especially at the reference resolution.
    "LV": (278, 774, 362, 799),
    "HP": (486, 767, 600, 787),
    "MP": (600, 767, 712, 787),
    # Add one reference pixel above/below the baseline.  At the live 2560x1440
    # client size the final digit of values such as 630498 sits on the crop
    # edge; the extra two pixels preserve it without materially widening the
    # recognition region.
    "EXP": (712, 766, 858, 788),
}

# Auxiliary regions used by the economy tracker.  They are deliberately kept
# outside STAT_PANEL_BOX so the existing HP/MP/EXP crops stay unchanged.
# Coordinates are measured from the same 1351x800 client screenshot used by
# the status panel.  The pickup feed is the right-side notification stack;
# SHORTCUT_SLOT_BOXES covers the two-row shortcut grid visible above the
# bottom action bar.  Users can leave unused slots blank in Settings.
PICKUP_FEED_BOX = (1080, 470, 1351, 665)
# Pickup toasts are brief and can shift with client/font scale.  Keep the
# narrow measured feed for cheap row OCR and a wider lower-right detector box.
PICKUP_WIDE_BOX = (760, 380, 1351, 720)
# The feed uses a stable 16px line rhythm in the reference client.  Cropping
# each row lets OCR stay in recognition-only mode (about an order of magnitude
# faster than detection+recognition) while preserving the row's y-position for
# duplicate-event filtering.
PICKUP_LINE_HEIGHT = 16
PICKUP_LINE_TOP_OFFSET = 8
PICKUP_LINE_BOXES = {
    str(index): (
        0,
        min(
            index * PICKUP_LINE_HEIGHT + PICKUP_LINE_TOP_OFFSET,
            PICKUP_FEED_BOX[3] - PICKUP_FEED_BOX[1] - 1,
        ),
        PICKUP_FEED_BOX[2] - PICKUP_FEED_BOX[0],
        min(
            index * PICKUP_LINE_HEIGHT + PICKUP_LINE_TOP_OFFSET + PICKUP_LINE_HEIGHT,
            PICKUP_FEED_BOX[3] - PICKUP_FEED_BOX[1],
        ),
    )
    for index in range(12)
}
SHORTCUT_BOX = (915, 650, 1080, 742)
SHORTCUT_SLOT_BOXES = {
    "1": (925, 659, 960, 695),
    "2": (960, 659, 997, 695),
    "3": (997, 659, 1034, 695),
    "4": (1034, 659, 1071, 695),
    "5": (925, 700, 960, 735),
    "6": (960, 700, 997, 735),
    "7": (997, 700, 1034, 735),
    "8": (1034, 700, 1071, 735),
}
AUXILIARY_BOXES = {
    "pickup": PICKUP_FEED_BOX,
    "pickup_wide": PICKUP_WIDE_BOX,
    "shortcut": SHORTCUT_BOX,
}

# The map title in the mini-map header and the job label beside the bottom
# status LV box are stable, low-frequency context signals.  They are captured
# separately from the 0.3s status fields and OCR'd only every few seconds.
# The class is on the first text line beside LV; the character name is on the
# line below it.  The older crop included both and frequently published the
# player name as the job.  Keep this crop on the class line only.
CONTEXT_BOXES = {
    # This is the second line under the mini-map header: the actual map name
    # (e.g. 第3軍營), not the first-line world/region label (維多利亞).
    "map": (0, 42, 130, 63),
    # A wider retry for clients where the tiny text is shifted left or the
    # recognition model drops the leading "第3" and returns only "軍/營".
    # Keep the focused crop above for the cheap first pass; the monitor only
    # uses this alternate crop when it needs to confirm the map.
    "map_wide": (0, 39, 180, 67),
    "job": (335, 768, 500, 795),
}


@dataclass(frozen=True)
class Box:
    left: int
    top: int
    right: int
    bottom: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)


@dataclass(frozen=True)
class RegionTransform:
    """Reference-viewport to current-client pixel transform."""

    client_size: tuple[int, int]
    scale: float
    offset_x: float
    offset_y: float

    def map_box(self, box: tuple[int, int, int, int]) -> Box:
        cw, ch = self.client_size
        left, top, right, bottom = box
        # Round edges independently, then clamp.  Clamping matters during a
        # resize transition where a previously valid crop can otherwise be one
        # pixel outside the new client frame.
        mapped_left = max(0, min(cw, round(self.offset_x + left * self.scale)))
        mapped_top = max(0, min(ch, round(self.offset_y + top * self.scale)))
        mapped_right = max(mapped_left + 1, min(cw, round(self.offset_x + right * self.scale)))
        mapped_bottom = max(mapped_top + 1, min(ch, round(self.offset_y + bottom * self.scale)))
        return Box(mapped_left, mapped_top, mapped_right, mapped_bottom)


def region_transform(client_size: tuple[int, int]) -> RegionTransform:
    """Build a transform for one actual captured client size.

    The game UI is rendered into a fixed-aspect viewport.  Fit that viewport
    uniformly instead of applying separate x/y stretch factors; when the
    window is wider than the reference, the unused horizontal space is a
    letterbox and the bottom status bar remains bottom-anchored.  This also
    avoids a one-frame crop drift while a window is being resized.
    """
    ref_w, ref_h = REFERENCE_CLIENT_SIZE
    client_w, client_h = client_size
    if ref_w <= 0 or ref_h <= 0 or client_w <= 0 or client_h <= 0:
        raise ValueError(f"invalid client size: {client_size!r}")

    scale = min(client_w / ref_w, client_h / ref_h)
    rendered_w = ref_w * scale
    rendered_h = ref_h * scale
    # Horizontal bars are centered.  The status panel is bottom anchored, so
    # any vertical spare pixels belong above the reference viewport.
    offset_x = (client_w - rendered_w) / 2
    offset_y = client_h - rendered_h
    return RegionTransform(client_size, scale, offset_x, offset_y)


def scale_box(box: tuple[int, int, int, int], client_size: tuple[int, int]) -> Box:
    """Map a reference crop box to the current client pixel coordinates."""
    return region_transform(client_size).map_box(box)


def scale_top_left_box(
    box: tuple[int, int, int, int], client_size: tuple[int, int]
) -> Box:
    """Scale a HUD box anchored to the client's top-left corner.

    The status/economy HUD is laid out inside the centered game viewport, so
    :func:`scale_box` correctly applies its horizontal letterbox offset. The
    mini-map is different: MapleStory pins it to the client origin. Applying
    the centered offset shifts the map title right and cuts off the leading
    ``第3`` on wide clients.
    """
    transform = region_transform(client_size)
    return RegionTransform(
        client_size=client_size,
        scale=transform.scale,
        offset_x=0,
        offset_y=0,
    ).map_box(box)
