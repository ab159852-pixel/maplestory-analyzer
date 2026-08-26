"""scale_box scaling math -- pure, no images/OCR."""
from maple_analyzer.regions import (
    FIELD_BOXES,
    REFERENCE_CLIENT_SIZE,
    STAT_PANEL_BOX,
    region_transform,
    scale_box,
    scale_shortcut_box,
    scale_top_left_box,
)


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
    # The game keeps the shortcut cell geometry tied to client width even when
    # the captured client is a few pixels shorter than the reference viewport.
    parent = scale_shortcut_box((915, 650, 1080, 742), (1368, 769))
    slot = scale_shortcut_box((960, 700, 997, 735), (1368, 769))

    assert parent.as_tuple() == (927, 617, 1094, 710)
    assert parent.left <= slot.left < slot.right <= parent.right
    assert parent.top <= slot.top < slot.bottom <= parent.bottom
