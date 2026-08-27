"""Crop-box definitions for the stat panel and its fields.

The boxes are measured in a reference game viewport, then mapped to the actual
client pixels reported by the capture backend on every frame.  The main status
panel uses the game's aspect-ratio transform.  The shortcut grid is a separate
bottom HUD layer: MapleStory scales that layer from the client width and keeps
it bottom-anchored even when the captured client is a little shorter than the
reference viewport.  Keeping those transforms explicit prevents the shortcut
quantities from drifting into the neighbouring cell on a different device.
"""
from __future__ import annotations

from dataclasses import dataclass

REFERENCE_CLIENT_SIZE = (1351, 800)  # size of samples/maple_story_ui.jpg

# A shortcut stack in MapleStory is displayed as a four-digit quantity at
# most.  Keep this domain rule next to the shortcut geometry so every OCR and
# accounting path can share the same hard boundary.
MAX_SHORTCUT_QUANTITY = 9_999
# Legacy boundary used only to classify optional HP/MP flash corroboration.
# Live accounting no longer rejects a larger drop: two matching quantity
# frames can provisionally charge the full difference, and a later stable
# quantity increase can reverse any OCR overcharge.  This keeps real 5-10
# bottle changes visible while still retaining the old diagnostic constant.
MAX_SHORTCUT_SINGLE_SAMPLE_DROP = 4

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

# The fixed HP/MP bar frames sit below the value text.  They are captured as
# lightweight visual signals, not passed through OCR: MapleStory briefly
# changes this frame/highlight when a potion is consumed.  Keeping these
# boxes separate from FIELD_BOXES preserves the exact four-field OCR contract
# used by the demo and regression fixtures.
BAR_BOXES = {
    # The numeric HP/MP text occupies the first few rows above the coloured
    # bar.  Keeping it out of this signal is important: a text redraw is not
    # the game's potion-flash effect and must never become a drink hint.
    "hp": (486, 785, 600, 800),
    "mp": (600, 785, 712, 800),
}

# Auxiliary regions used by the economy tracker.  They are deliberately kept
# outside STAT_PANEL_BOX so the existing HP/MP/EXP crops stay unchanged.
# Coordinates are measured from the same 1351x800 client screenshot used by
# the status panel.  The pickup feed is the right-side notification stack;
# SHORTCUT_BOX is the visible outer frame of the 4x2 shortcut grid, not the
# larger area around it.  The previous implementation used a 165x92 box
# around a frame that is only 147x77 at the reference size.  That extra
# padding made the eight crops overlap the frame and, after scaling, drift
# into their neighbours.  Keep the frame and the eight interior cells in one
# calibrated coordinate system.
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
PICKUP_LINE_PADDING = 3
PICKUP_LINE_BOXES = {
    str(index): (
        0,
        min(
            max(
                0,
                index * PICKUP_LINE_HEIGHT + PICKUP_LINE_TOP_OFFSET - PICKUP_LINE_PADDING,
            ),
            PICKUP_FEED_BOX[3] - PICKUP_FEED_BOX[1] - 1,
        ),
        PICKUP_FEED_BOX[2] - PICKUP_FEED_BOX[0],
        min(
            index * PICKUP_LINE_HEIGHT
            + PICKUP_LINE_TOP_OFFSET
            + PICKUP_LINE_HEIGHT
            + PICKUP_LINE_PADDING,
            PICKUP_FEED_BOX[3] - PICKUP_FEED_BOX[1],
        ),
    )
    for index in range(12)
}
# Measured from the real frame in samples/maple_story_ui.jpg.  Right/bottom
# are exclusive PIL crop edges.
SHORTCUT_BOX = (925, 659, 1072, 736)

# Cell boxes use the midpoint of each real separator as their boundary.  The
# crop therefore contains the whole cell (including the outlined last digit)
# but never enters the next cell.  The separator itself is harmless chrome;
# trimming it out was the previous cause of values such as 1570 becoming 57.
SHORTCUT_SLOT_BOXES = {
    "1": (925, 659, 963, 700),
    "2": (963, 659, 998, 700),
    "3": (998, 659, 1033, 700),
    "4": (1033, 659, 1072, 700),
    "5": (925, 700, 963, 736),
    "6": (963, 700, 998, 736),
    "7": (998, 700, 1033, 736),
    "8": (1033, 700, 1072, 736),
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

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

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


def shortcut_transform(client_size: tuple[int, int]) -> RegionTransform:
    """Build the transform for the bottom-right shortcut grid.

    The shortcut/action HUD is rendered as a width-scaled layer, rather than
    being fitted to the full captured aspect ratio.  In particular, a
    1368x769 client keeps nearly the reference shortcut cell size; applying
    ``region_transform`` there shrinks the cells and shifts the first column
    left far enough for OCR to read a neighbouring quantity.  Width scaling
    preserves the grid geometry and bottom anchoring while still adapting to
    genuinely wider clients such as 1920px captures.
    """
    ref_w, ref_h = REFERENCE_CLIENT_SIZE
    client_w, client_h = client_size
    if ref_w <= 0 or ref_h <= 0 or client_w <= 0 or client_h <= 0:
        raise ValueError(f"invalid client size: {client_size!r}")
    scale = client_w / ref_w
    rendered_h = ref_h * scale
    return RegionTransform(
        client_size=client_size,
        scale=scale,
        # The bottom HUD is centred horizontally in the client layer.  Since
        # the layer's logical width equals the reference width, this is zero
        # at the reference size and naturally tracks a wider client.
        offset_x=0,
        offset_y=client_h - rendered_h,
    )


def scale_shortcut_box(
    box: tuple[int, int, int, int], client_size: tuple[int, int]
) -> Box:
    """Map a shortcut-grid box to the current client pixels."""
    return shortcut_transform(client_size).map_box(box)


def shortcut_slot_boxes_for_parent(parent: Box) -> dict[str, Box]:
    """Map the calibrated eight cells into an already-cropped grid frame.

    Live capture may first locate the actual frame from its border pixels.  In
    that case the frame can be a few pixels larger/smaller than the fallback
    transform.  Deriving every cell from that one parent keeps all columns and
    rows aligned and prevents independent rounding from creating slanted
    boundaries.
    """
    frame_left, frame_top, frame_right, frame_bottom = SHORTCUT_BOX
    frame_width = max(1, frame_right - frame_left)
    frame_height = max(1, frame_bottom - frame_top)
    result: dict[str, Box] = {}
    for slot, raw in SHORTCUT_SLOT_BOXES.items():
        left, top, right, bottom = raw
        mapped_left = parent.left + round((left - frame_left) * parent.width / frame_width)
        mapped_top = parent.top + round((top - frame_top) * parent.height / frame_height)
        mapped_right = parent.left + round((right - frame_left) * parent.width / frame_width)
        mapped_bottom = parent.top + round((bottom - frame_top) * parent.height / frame_height)
        result[slot] = Box(
            max(parent.left, min(parent.right - 1, mapped_left)),
            max(parent.top, min(parent.bottom - 1, mapped_top)),
            max(parent.left + 1, min(parent.right, mapped_right)),
            max(parent.top + 1, min(parent.bottom, mapped_bottom)),
        )
    return result
