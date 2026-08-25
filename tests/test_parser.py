"""Pure regex/normalization tests for parser.py -- no OCR, no images. Covers
the field text -> StatSnapshot logic and the OCR-noise cases documented in
parser.py's module docstring (dropped '.', dropped ']', space-for-dot)."""
from maple_analyzer.parser import parse_fields


def test_clean_fields():
    snap = parse_fields({
        "LV": "LV.44", "HP": "HP[377/824]", "MP": "MP[1663/2816]", "EXP": "EXP162950[38.05%]",
    })
    assert snap.level == 44
    assert (snap.hp_cur, snap.hp_max) == (377, 824)
    assert (snap.mp_cur, snap.mp_max) == (1663, 2816)
    assert snap.exp_cur == 162950
    assert snap.exp_pct == 38.05


def test_missing_dot_in_pct_normalized():
    # Bare 3-4 digit run before '%' implies 2 decimal places.
    snap = parse_fields({"EXP": "EXP162950[3805%]"})
    assert snap.exp_pct == 38.05


def test_missing_bracket_is_now_unreadable():
    """This used to parse: the bracket was treated as decoration, so a read
    that lost it still yielded a number. That tolerance is what let
    'EXP357041183.37%]' merge 357041 and 183 into 357,041,183, and
    'EXP101332182' book +101,322,049 of phantom gain.

    Requiring the bracket costs 0.4% of reads (51 of 12,384 measured live) and
    nearly all of those are garbage.

    The percentage goes with it: _find_exp only looks for pct after a
    successful cur match, so a broken structure discards both. That coupling
    is deliberate -- if the field doesn't look like `cur[pct%]`, neither
    number in it is trustworthy."""
    snap = parse_fields({"EXP": "EXP162950 3805%"})
    assert snap.exp_cur is None
    assert snap.exp_pct is None

    # the same reading with its bracket intact still works
    ok = parse_fields({"EXP": "EXP162950[3805%]"})
    assert (ok.exp_cur, ok.exp_pct) == (162950, 38.05)


def test_space_for_dot_in_pct():
    snap = parse_fields({"EXP": "EXP162950[63 14%]"})
    assert snap.exp_pct == 63.14


def test_short_digit_run_before_percent_not_matched():
    # A bare 1-2 digit run is deliberately ambiguous, per parser.py -- should
    # not be picked up as a percentage.
    snap = parse_fields({"EXP": "EXP162950[5%]"})
    assert snap.exp_pct is None


def test_missing_field_is_none():
    snap = parse_fields({})
    assert snap.level is None
    assert snap.hp_cur is None and snap.hp_max is None
    assert snap.mp_cur is None and snap.mp_max is None
    assert snap.exp_cur is None and snap.exp_pct is None


def test_garbage_text_does_not_raise():
    snap = parse_fields({"LV": "??", "HP": "", "MP": "garbage###", "EXP": "not an exp string"})
    assert snap.level is None
    assert snap.hp_cur is None
    assert snap.mp_cur is None
    assert snap.exp_cur is None


def test_hp_mp_case_insensitive_label():
    snap = parse_fields({"HP": "hp[10/20]", "MP": "mp[5/30]"})
    assert (snap.hp_cur, snap.hp_max) == (10, 20)
    assert (snap.mp_cur, snap.mp_max) == (5, 30)


def test_isolated_fields_recover_when_ascii_label_is_dropped():
    snap = parse_fields({
        "HP": "[10/20]",
        "MP": "5/30",
        "EXP": "123456[12.34%]",
        "LV": "44",
    })
    assert (snap.hp_cur, snap.hp_max) == (10, 20)
    assert (snap.mp_cur, snap.mp_max) == (5, 30)
    assert (snap.exp_cur, snap.exp_pct) == (123456, 12.34)
    assert snap.level == 44


def test_exp_pct_separator_read_as_colon():
    """Live capture (2026-08-17) showed the decimal point OCR'd as a colon in
    37 of 235 ticks -- each one silently lost exp_pct, and with it the EXP%
    readout and the level-up ETA."""
    snap = parse_fields({"EXP": "EXP 321675[75:11%] "})
    assert snap.exp_cur == 321675
    assert snap.exp_pct == 75.11


def test_exp_thousands_separator_is_accepted():
    snap = parse_fields({"EXP": "EXP 1,902,660[9.73%]"})
    assert snap.exp_cur == 1902660
    assert snap.exp_pct == 9.73
