"""Crop-box definitions for the stat panel and its fields.

The boxes are measured in the complete reference window image, then mapped to
the client pixels reported by the live capture backend on every frame. The
reference images include the native Windows title bar; live WGC/PrintWindow
frames do not. The main status panel uses the game's aspect-ratio transform.
The shortcut grid is a separate bottom HUD layer: MapleStory scales that
layer from the client width and keeps it bottom-anchored even when the
captured client is a little shorter than the reference viewport. Keeping
those transforms explicit prevents the shortcut quantities from drifting into
the neighbouring cell on a different device.
"""
from __future__ import annotations

from dataclasses import dataclass

# The bundled reference captures are complete top-level window screenshots,
# not client-only frames: their first 35-ish pixels are the Windows title bar.
# Keep the old public name for compatibility with the static-image tests, but
# make the coordinate contract explicit. Live WGC/PrintWindow/mss captures
# are client-only and must use the window->client helpers below.
REFERENCE_WINDOW_SIZE = (1351, 800)  # size of samples/maple_story_ui.jpg
REFERENCE_CLIENT_SIZE = REFERENCE_WINDOW_SIZE  # legacy public alias
# The reference screenshot contains a 36px native title bar.  Live capture
# backends return only the game client, so top-left-pinned UI (the mini-map)
# must first be converted into this client coordinate system and only then be
# scaled to the current frame.  Scaling the complete window and subtracting
# the *current* title bar afterwards drifts the crop downward at 2K.
REFERENCE_CLIENT_TOP = 36
REFERENCE_GAME_CLIENT_SIZE = (
    REFERENCE_WINDOW_SIZE[0],
    REFERENCE_WINDOW_SIZE[1] - REFERENCE_CLIENT_TOP,
)

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

# Whole stat panel, in absolute pixels at REFERENCE_WINDOW_SIZE. Grabbed once per
# tick with a single mss.grab() call; FIELD_BOXES below are sliced out of it
# in-memory (no extra screen captures).
STAT_PANEL_BOX = (260, 758, 900, 800)  # (left, top, right, bottom)

# Per-field boxes, same reference-window frame as STAT_PANEL_BOX, generously padded
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
# Coordinates are measured from the same 1351x800 reference-window screenshot used by
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
# The feed uses a stable 16px line rhythm in the reference window. Cropping
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
    # The mini-map has two text rows below its tab strip.  The first row is
    # the region/world (for example 維多利亞); the second row is the actual
    # map (for example 魔法森林北部 or 第3軍營).  The previous y=42..63
    # crop was still on the header/first row, which is why the live map name
    # was either wrong or became a noisy partial fragment.
    "map": (0, 75, 145, 101),
    # Keep the second row plus a small horizontal/vertical safety margin for
    # DPI rounding and longer map names.  This remains a separate retry, not
    # part of the high-frequency status/potion OCR path.
    "map_wide": (0, 70, 200, 110),
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
    """Reference-frame to current-image pixel transform."""

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
    """Build a transform for one actual target-image size.

    The game UI is rendered into a fixed-aspect viewport.  Fit that viewport
    uniformly instead of applying separate x/y stretch factors; when the
    window is wider than the reference, the unused horizontal space is a
    letterbox and the bottom status bar remains bottom-anchored.  This also
    avoids a one-frame crop drift while a window is being resized.
    """
    ref_w, ref_h = REFERENCE_WINDOW_SIZE
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
    """Map a reference crop box to the current image pixel coordinates."""
    return region_transform(client_size).map_box(box)


def scale_top_left_box(
    box: tuple[int, int, int, int], client_size: tuple[int, int]
) -> Box:
    """Scale a HUD box anchored to the target image's top-left corner.

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


def _shift_box_to_client(
    box: Box,
    client_size: tuple[int, int],
    client_offset: tuple[int, int],
) -> Box:
    """Translate a full-window box into a client-only image and clamp it.

    ``client_offset`` is the physical-pixel position of the client origin
    inside the captured top-level window (normally the title-bar height plus
    any invisible frame inset). Keeping this step after the reference
    viewport transform is important: title bars and DPI borders are measured
    in the actual window, not guessed from a fixed 35-pixel constant.
    """
    client_width, client_height = client_size
    offset_x, offset_y = client_offset
    if client_width <= 0 or client_height <= 0:
        raise ValueError(f"invalid client size: {client_size!r}")

    left = round(box.left - offset_x)
    top = round(box.top - offset_y)
    right = round(box.right - offset_x)
    bottom = round(box.bottom - offset_y)
    left = max(0, min(client_width - 1, left))
    top = max(0, min(client_height - 1, top))
    right = max(left + 1, min(client_width, right))
    bottom = max(top + 1, min(client_height, bottom))
    return Box(left, top, right, bottom)


def _map_window_box_to_client(
    box: tuple[int, int, int, int],
    client_size: tuple[int, int],
    window_size: tuple[int, int],
    client_offset: tuple[int, int],
    *,
    top_left: bool = False,
    shortcut: bool = False,
) -> Box:
    """Map a reference full-window box into a live client-only frame.

    ``window_size`` is the actual top-level HWND frame size, while
    ``client_size`` is the image returned by the capture backend. The
    reference boxes were measured in the former coordinate system. For the
    mini-map, ``top_left`` removes viewport letterboxing because the game pins
    that HUD to the client origin. The shortcut grid uses its own
    width-scaled/bottom-anchored transform.
    """
    transform = shortcut_transform(window_size) if shortcut else region_transform(window_size)
    if top_left:
        transform = RegionTransform(
            client_size=window_size,
            scale=transform.scale,
            offset_x=0,
            offset_y=0,
        )
    window_box = transform.map_box(box)
    return _shift_box_to_client(window_box, client_size, client_offset)


def scale_window_box_to_client(
    box: tuple[int, int, int, int],
    client_size: tuple[int, int],
    window_size: tuple[int, int],
    client_offset: tuple[int, int] = (0, 0),
) -> Box:
    """Map a centered/bottom-anchored full-window box to client pixels."""
    return _map_window_box_to_client(
        box, client_size, window_size, client_offset
    )


def scale_window_top_left_box_to_client(
    box: tuple[int, int, int, int],
    client_size: tuple[int, int],
    window_size: tuple[int, int],
    client_offset: tuple[int, int] = (0, 0),
) -> Box:
    """Map a client-origin-pinned reference-window box to live client pixels.

    ``box`` was measured in a screenshot that includes the reference title
    bar, while WGC/PrintWindow already return client-only pixels.  Convert the
    reference Y coordinates first, then scale against the actual client.  The
    real ``window_size``/``client_offset`` remain part of the public signature
    for consistency with the other live mappers, but must not be applied a
    second time to a client-only frame.
    """
    del window_size, client_offset
    left, top, right, bottom = box
    client_box = (
        left,
        max(0, top - REFERENCE_CLIENT_TOP),
        right,
        max(1, bottom - REFERENCE_CLIENT_TOP),
    )
    ref_width, ref_height = REFERENCE_GAME_CLIENT_SIZE
    client_width, client_height = client_size
    if min(ref_width, ref_height, client_width, client_height) <= 0:
        raise ValueError(f"invalid client size: {client_size!r}")
    scale = min(client_width / ref_width, client_height / ref_height)
    return RegionTransform(
        client_size=client_size,
        scale=scale,
        offset_x=0,
        offset_y=0,
    ).map_box(client_box)


def scale_window_shortcut_box_to_client(
    box: tuple[int, int, int, int],
    client_size: tuple[int, int],
    window_size: tuple[int, int],
    client_offset: tuple[int, int] = (0, 0),
) -> Box:
    """Map a bottom-right shortcut box from the full window to client pixels."""
    return _map_window_box_to_client(
        box, client_size, window_size, client_offset, shortcut=True
    )


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
    ref_w, ref_h = REFERENCE_WINDOW_SIZE
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
