"""scale_box scaling math -- pure, no images/OCR."""
from maple_analyzer.regions import (
    AUXILIARY_BOXES,
    Box,
    FIELD_BOXES,
    REFERENCE_CLIENT_SIZE,
    SHORTCUT_BOX,
    SHORTCUT_SLOT_BOXES,
    STAT_PANEL_BOX,
    scale_box,
    region_transform,
    scale_shortcut_box,
    scale_top_left_box,
    shortcut_slot_boxes_for_parent,
)
from maple_analyzer.capture import _pickup_boxes_for_client, detect_shortcut_frame
from conftest import SAMPLE_IMAGE, SAMPLE_IMAGE_1920
from PIL import Image
from maple_analyzer.ocr import _shortcut_quantity_strip


def test_identity_scale_at_reference_size():
    for box in [STAT_PANEL_BOX, *FIELD_BOXES.values()]:
        assert scale_box(box, REFERENCE_CLIENT_SIZE).as_tuple() == box


def test_scales_proportionally():
    ref_w, ref_h = REFERENCE_CLIENT_SIZE
    box = scale_box(STAT_PANEL_BOX, (ref_w * 2, ref_h * 2))
    assert box.as_tuple() == tuple(c * 2 for c in STAT_PANEL_BOX)


def test_field_boxes_stay_within_panel_at_reference_size():
    panel = scale_box(STAT_PANEL_BOX, REFERENCE_CLIENT_SIZE)
    for name, box in FIELD_BOXES.items():
        b = scale_box(box, REFERENCE_CLIENT_SIZE)
        assert panel.left <= b.left and b.right <= panel.right, name
        assert panel.top <= b.top and b.bottom <= panel.bottom, name


def test_scale_at_known_working_resolutions():
    # Confirmed working live per handover notes: 1366x768 and 1920x1080.
    for client_size in [(1366, 768), (1920, 1080)]:
        panel = scale_box(STAT_PANEL_BOX, client_size)
        assert panel.left < panel.right and panel.top < panel.bottom
        for box in FIELD_BOXES.values():
            b = scale_box(box, client_size)
            assert b.left < b.right and b.top < b.bottom


def test_wide_client_uses_uniform_scale_and_horizontal_letterbox():
    transform = region_transform((1920, 1077))
    assert transform.offset_x > 0
    assert transform.offset_y == 0
    assert transform.scale == 1077 / REFERENCE_CLIENT_SIZE[1]

    panel = scale_box(STAT_PANEL_BOX, (1920, 1077))
    assert panel.bottom == 1077
    assert panel.left > 0


def test_top_left_box_does_not_apply_viewport_letterbox():
    box = (44, 42, 119, 63)
    client_size = (1920, 1077)
    transform = region_transform(client_size)

    centered = scale_box(box, client_size)
    anchored = scale_top_left_box(box, client_size)

    assert centered.left > anchored.left
    assert anchored.left == round(box[0] * transform.scale)
    assert anchored.top == round(box[1] * transform.scale)


def test_shortcut_grid_is_width_scaled_and_bottom_anchored():
    # The fallback transform remains deterministic before the border detector
    # has a frame to measure.  The eight cells must still be inside one parent.
    parent = scale_shortcut_box(SHORTCUT_BOX, (1368, 769))
    slot = scale_shortcut_box(SHORTCUT_SLOT_BOXES["2"], (1368, 769))

    assert parent.as_tuple() == (937, 626, 1085, 704)
    assert parent.left <= slot.left < slot.right <= parent.right
    assert parent.top <= slot.top < slot.bottom <= parent.bottom


def test_shortcut_cells_are_aligned_and_non_overlapping():
    for parent in (
        Box(0, 0, SHORTCUT_BOX[2] - SHORTCUT_BOX[0], SHORTCUT_BOX[3] - SHORTCUT_BOX[1]),
        Box(0, 0, 147, 77),
        Box(0, 0, 206, 108),
    ):
        slots = shortcut_slot_boxes_for_parent(parent)
        assert set(slots) == set(SHORTCUT_SLOT_BOXES)
        assert all(
            0 <= box.left < box.right <= parent.width
            and 0 <= box.top < box.bottom <= parent.height
            for box in slots.values()
        )
        for row in (("1", "2", "3", "4"), ("5", "6", "7", "8")):
            assert [slots[slot].top for slot in row].count(slots[row[0]].top) == 4
            assert [slots[slot].bottom for slot in row].count(slots[row[0]].bottom) == 4
            for left, right in zip(row, row[1:]):
                assert slots[left].right <= slots[right].left


def test_shortcut_quantity_strip_keeps_the_complete_four_digit_width():
    # The crop is already bounded by separator midpoints.  The OCR strip must
    # retain the rightmost pixels because the fourth digit can reach that edge.
    for width, height in ((38, 41), (35, 41), (39, 36), (53, 58)):
        strip = _shortcut_quantity_strip(Image.new("RGB", (width, height)))
        assert strip.width == width
        assert strip.height == height - round(height * 0.48)


def test_shortcut_frame_detector_tracks_real_reference_sizes():
    for path in (SAMPLE_IMAGE, SAMPLE_IMAGE_1920):
        image = Image.open(path).convert("RGB")
        expected = scale_shortcut_box(SHORTCUT_BOX, image.size)
        frame = detect_shortcut_frame(image, expected)
        assert frame.width >= round(expected.width * 0.65)
        assert frame.height >= round(expected.height * 0.60)
        assert 0 <= frame.left < frame.right <= image.width
        assert 0 <= frame.top < frame.bottom <= image.height


def test_wide_client_extends_pickup_feed_to_the_actual_client_edge():
    client = (1368, 768)
    boxes = _pickup_boxes_for_client(client)
    pickup = scale_box(boxes["pickup"], client)
    reference = scale_box(AUXILIARY_BOXES["pickup"], client)

    assert pickup.right == client[0]
    assert pickup.right > reference.right
