"""Thin wrapper around RapidOCR (PP-OCR recognition, ONNX/CPU).

See VERSIONS.md for why rapidocr-onnxruntime stands in for the spec's originally
named PP-OCRv6-tiny ONNX model -- same OCR family, bundled models, no manual
model-file wiring.

Recognition-only, not detection+recognition: benchmarked live against the real
game (2026-08-17), detection (finding text regions in an image) was ~600-680ms
per call -- the entire OCR bottleneck the rest of the pipeline was tuned around.
Recognition alone (reading a pre-cropped, known-to-contain-one-line-of-text
image) was ~15ms. Since regions.py's FIELD_BOXES already pins down exactly
where each field's text is, running detection to *re-discover* that on every
tick was pure waste -- see capture.py's grab_fields() for the cropping side of
this change.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
import re
from collections import Counter
from typing import Any, Iterable

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from .regions import SHORTCUT_BOX, SHORTCUT_SLOT_BOXES


@dataclass(frozen=True)
class OcrLine:
    text: str
    y: float | None = None
    x: float | None = None
    left: float | None = None
    right: float | None = None
    score: float | None = None


class StatPanelOcr:
    def __init__(self) -> None:
        # Keep the heavyweight ONNX/RapidOCR import out of module import time.
        # The HUD can paint immediately, then OverlayApp constructs this class
        # on its background loader once the user already has a responsive
        # window.
        from rapidocr_onnxruntime import RapidOCR

        self._engine = RapidOCR()

    def read_field(self, image: Image.Image) -> str:
        """Recognition-only OCR on a small pre-cropped single-line field crop.

        This remains the cheap compatibility API used by context OCR and test
        doubles.  The live status path calls :meth:`read_fields`, which knows
        which field it is reading and can validate/retry only the fields whose
        first recognition is structurally weak.
        """
        text, _records = self._read_once(image)
        # Tiny EXP glyphs occasionally arrive with a space inserted inside the
        # current value (``EXP100 100[01.00%]``) or with the opening bracket
        # dropped.  Retry only for that structurally suspicious EXP shape; the
        # normal 300ms loop keeps its single fast recognition call.
        if _looks_like_exp_but_is_malformed(text):
            # A redraw can leave two rows of the old EXP bar at the bottom of
            # the padded crop. Removing those rows often restores the opening
            # bracket; keep the enlarged view as a second fallback.
            variants = []
            if image.height > 4:
                variants.append(image.crop((0, 0, image.width, image.height - 2)))
            variants.append(self._enlarged(image))
            for variant in variants:
                retry_text, _retry_records = self._read_once(variant)
                if _looks_like_exp(retry_text):
                    return retry_text
        return text

    def read_fields(self, images: dict[str, Image.Image]) -> dict[str, str]:
        """Read the four status fields with cheap, format-aware recovery.

        Most frames take exactly one recognition call per field.  A second
        pass is made only when the text is missing its expected structure or
        RapidOCR reports low confidence.  The retry uses an enlarged,
        contrast-normalized crop; if both passes are usable, the candidate
        with the strongest confidence is selected, while repeated parsed
        values are preferred over a one-off disagreement.

        Keeping this inside the existing RapidOCR instance preserves the fast
        recognition-only path.  Replacing it with detection OCR on every tick
        would be substantially slower and would not solve the usual tiny-font
        failure mode.
        """
        return {
            name: self._read_typed_field(name, image)
            for name, image in images.items()
        }

    def _read_typed_field(self, field: str, image: Image.Image) -> str:
        first_text, first_records = self._read_once(image)
        first_valid = _field_is_valid(field, first_text)
        first_confidence = _confidence(first_records)

        # A high-confidence, structurally valid read is the common path.  Do
        # not spend another ONNX call on every 300ms tick just to marginally
        # change an already trustworthy result.
        if first_valid and (first_confidence is None or first_confidence >= 0.78):
            return first_text

        candidates = [(first_text, first_records)]
        # Two variants cover the common causes without invoking detection:
        # interpolation recovers tiny glyph edges, while grayscale/autocontrast
        # reduces coloured UI-bar noise around white digits.
        for variant in self._retry_variants(image):
            candidates.append(self._read_once(variant))

        usable = [
            (text, records)
            for text, records in candidates
            if _field_is_valid(field, text)
        ]
        if not usable:
            # Preserve the raw first result so the parser can carry forward the
            # last known value instead of turning a transient miss into an
            # exception.
            return first_text

        signatures: dict[object, int] = {}
        for text, _records in usable:
            signature = _field_signature(field, text)
            signatures[signature] = signatures.get(signature, 0) + 1
        return max(
            usable,
            key=lambda candidate: (
                signatures[_field_signature(field, candidate[0])],
                _confidence(candidate[1]) or 0.0,
            ),
        )[0]

    def _read_once(self, image: Image.Image) -> tuple[str, list[OcrLine]]:
        records = self._recognize(image, use_det=False)
        return " ".join(record.text for record in records), records

    @staticmethod
    def _enlarged(image: Image.Image) -> Image.Image:
        resampling = getattr(Image, "Resampling", Image)
        return image.resize(
            (max(1, image.width * 2), max(1, image.height * 2)),
            resampling.LANCZOS,
        )

    @classmethod
    def _retry_variants(cls, image: Image.Image) -> tuple[Image.Image, ...]:
        enlarged = cls._enlarged(image)
        gray = ImageOps.autocontrast(ImageOps.grayscale(enlarged))
        gray = ImageEnhance.Contrast(gray).enhance(1.35)
        gray = gray.filter(ImageFilter.SHARPEN)
        return enlarged, gray

    def read_lines(self, image: Image.Image) -> list[OcrLine]:
        """Detection+recognition for a region containing multiple text lines."""
        return self._recognize(image, use_det=True)

    def read_slot_count(self, image: Image.Image) -> int | None:
        """Read a shortcut quantity with numeric-focused OCR consensus.

        Shortcut cells contain both item/count glyphs and keyboard labels such
        as Shift or Ins. A single pass often reads the label or a partial
        digit, so accept a value only when two independently preprocessed
        views agree.
        """
        resampling = getattr(Image, "Resampling", Image)
        enlarged = image.resize(
            (max(1, image.width * 4), max(1, image.height * 4)),
            resampling.LANCZOS,
        )
        lower_right = image.crop((
            image.width // 3,
            image.height // 2,
            image.width,
            image.height,
        ))
        lower_right_large = lower_right.resize(
            (max(1, lower_right.width * 5), max(1, lower_right.height * 5)),
            resampling.LANCZOS,
        )
        gray = ImageOps.autocontrast(ImageOps.grayscale(enlarged))
        gray = ImageEnhance.Contrast(gray).enhance(1.5)
        def extract(text: str) -> int | None:
            matches = re.findall(r"\d[\d,]*", text.replace(" ", ""))
            if not matches:
                return None
            try:
                return int(matches[-1].replace(",", ""))
            except ValueError:
                return None

        # The common path stays one OCR call per slot.  Only a crop that did
        # not expose any number pays for the recovery views; this matters when
        # all eight slots are being observed automatically.
        first_text, _records = self._read_once(image)
        first_value = extract(first_text)
        if first_value is not None:
            return first_value

        candidates: list[int] = []
        for variant in (enlarged, lower_right, lower_right_large, gray):
            text, _records = self._read_once(variant)
            value = extract(text)
            if value is not None:
                candidates.append(value)
        if not candidates:
            return None
        value, confirmations = Counter(candidates).most_common(1)[0]
        return value if confirmations >= 2 else None

    def read_shortcut_counts(
        self,
        image: Image.Image,
        required_slots: Iterable[str] | None = None,
        blue_slots: Iterable[str] | None = None,
    ) -> dict[str, int]:
        """Read all visible shortcut quantities from the parent bar.

        At non-reference client sizes a quantity can extend across the small
        per-cell crop boundary (for example ``1180`` becomes ``118`` and the
        next cell is read as ``7465``).  Detection on the complete bar retains
        the glyph box; its horizontal position lets us split a merged run and
        assign the pieces back to slots without guessing from a single cell.
        """
        required = set(required_slots or ())
        blue = set(blue_slots or ())
        # Configured potion slots have a known cell location.  Recognition on
        # the small lower quantity strip is both much faster than detection on
        # the whole bar and avoids adjacent cells being merged into one OCR
        # run.  The wider detection path remains below as a recovery/fallback
        # for custom callers that ask for slots which the fast crop misses.
        if required:
            fast_counts = self._read_shortcut_slot_counts(image, required, blue)
            if required.issubset(fast_counts):
                return fast_counts
        else:
            fast_counts = {}

        counts = self._shortcut_counts_from_records(image, self.read_lines(image))
        counts.update(fast_counts)
        missing = required - counts.keys()
        if not missing:
            return counts

        # Detection occasionally loses only the small glyph run for one slot.
        # Retry enhanced full-bar images only for the missing configured slots;
        # blank/unconfigured slots do not force an expensive extra detection
        # pass on every auxiliary scan.
        candidates: dict[str, list[int]] = {slot: [] for slot in missing}
        for variant in self._retry_variants(image):
            retry = self._shortcut_counts_from_records(variant, self.read_lines(variant))
            for slot in missing:
                if slot in retry:
                    candidates[slot].append(retry[slot])
        for slot, values in candidates.items():
            if values:
                # Prefer the widest numeric interpretation when a sharpened
                # view recovers leading digits that the original missed.
                counts[slot] = max(values, key=lambda value: (len(str(value)), value))
        if required:
            # A full-bar recovery can see neighboring/blank shortcut cells.
            # When the caller names configured slots, never leak those other
            # cells into potion accounting (e.g. slot 8's 915 into slot 7).
            return {slot: counts[slot] for slot in required if slot in counts}
        return counts

    def _read_shortcut_slot_counts(
        self,
        image: Image.Image,
        required_slots: Iterable[str],
        blue_slots: Iterable[str] = (),
    ) -> dict[str, int]:
        """Read configured cells without running full-bar text detection.

        The quantity is rendered at the bottom of each cell and can touch the
        right edge of the reference crop at non-100% DPI.  Expand the crop
        slightly to the right, and for columns after the first add a small
        left inset so the previous cell's last digit is not included.
        """
        parent_w, parent_h = image.size
        ref_w = SHORTCUT_BOX[2] - SHORTCUT_BOX[0]
        ref_h = SHORTCUT_BOX[3] - SHORTCUT_BOX[1]
        scale_x = parent_w / ref_w
        scale_y = parent_h / ref_h
        blue_slot_ids = {str(slot_id) for slot_id in blue_slots}
        counts: dict[str, int] = {}
        for slot_id in required_slots:
            box = SHORTCUT_SLOT_BOXES.get(str(slot_id))
            if box is None:
                continue
            try:
                column = (int(slot_id) - 1) % 4
            except (TypeError, ValueError):
                column = 1
            # The quantity is right-aligned in the cell.  A four-pixel inset
            # clips the leading ``1`` in values such as 115 at the current
            # client scale, turning the blue-water count into 15.  No inset
            # includes the previous cell's trailing ``1`` and turns it into
            # 1115.  Five reference pixels keeps the whole current quantity
            # while excluding that neighboring glyph.
            # The second cell in each row has a one-pixel render seam that
            # makes the standard inset turn a 3 into a 9 (for example
            # 1239 -> 1299).  A smaller inset and one-pixel lower baseline
            # keeps the complete outlined digit while excluding the previous
            # cell.  Slot 6 is the usual HP-water position.
            left_pad = 2 if column == 1 else (5 if column else 0)
            right_pad = 8 if column < 3 else 0
            crop = image.crop((
                max(0, round((box[0] - SHORTCUT_BOX[0] + left_pad) * scale_x)),
                # The quantity glyphs begin around +10 in the reference cell.
                # Starting lower (+14/+16) can still include the bottom of the
                # keyboard label and produce a false leading digit (for
                # example 1343 becomes 11349).  The +10 crop keeps the full
                # quantity while excluding the label; read_slot_count also
                # tolerates the small outlined-border zero at the left edge.
                max(0, round((box[1] - SHORTCUT_BOX[1] + (9 if column == 1 else 10)) * scale_y)),
                min(parent_w, round((box[2] - SHORTCUT_BOX[0] + right_pad) * scale_x)),
                min(parent_h, round((box[3] - SHORTCUT_BOX[1] + 2) * scale_y)),
            ))
            count = self.read_slot_count(crop)
            # The blue potion icon can overlap the left side of the quantity
            # strip.  RapidOCR then either drops the leading digit (91 -> 9)
            # or joins it with the neighboring cell (91 -> 915).  A targeted
            # colour-neutralized pass is more trustworthy for that cell than
            # either ambiguous result.  White HP counts keep the cheap path.
            if str(slot_id) in blue_slot_ids or _looks_like_blue_shortcut_cell(crop):
                blue_count = _read_blue_shortcut_count(
                    self,
                    image,
                    box,
                    scale_x,
                    scale_y,
                )
                if blue_count is not None:
                    count = blue_count
            if count is not None:
                counts[str(slot_id)] = count
        return counts

    @staticmethod
    def _shortcut_counts_from_records(
        image: Image.Image,
        records: list[OcrLine],
    ) -> dict[str, int]:
        if not records:
            return {}

        parent_w, parent_h = image.size
        ref_w = SHORTCUT_BOX[2] - SHORTCUT_BOX[0]
        ref_h = SHORTCUT_BOX[3] - SHORTCUT_BOX[1]
        scale_x = parent_w / ref_w
        scale_y = parent_h / ref_h
        centers_x = [
            ((box[0] + box[2]) / 2 - SHORTCUT_BOX[0]) * scale_x
            for box in SHORTCUT_SLOT_BOXES.values()
        ]
        centers_y = [
            ((SHORTCUT_SLOT_BOXES["1"][1] + SHORTCUT_SLOT_BOXES["1"][3]) / 2 - SHORTCUT_BOX[1]) * scale_y,
            ((SHORTCUT_SLOT_BOXES["5"][1] + SHORTCUT_SLOT_BOXES["5"][3]) / 2 - SHORTCUT_BOX[1]) * scale_y,
        ]

        # The game may add a small left inset at a different render scale.
        # Use the first-row keyboard label as an anchor when available.
        first_label = [
            record for record in records
            if record.x is not None
            and record.y is not None
            and record.y < parent_h * 0.42
            and not re.search(r"\d", record.text)
        ]
        x_offset = 0.0
        if first_label:
            anchor = min(first_label, key=lambda record: abs((record.x or 0.0) - centers_x[0]))
            x_offset = (anchor.x or 0.0) - centers_x[0]
        centers_x = [center + x_offset for center in centers_x]

        counts: dict[str, int] = {}
        for record in records:
            if record.x is None or record.y is None:
                continue
            # Keep only a clean numeric OCR run.  Keyboard labels such as
            # ``1元`` are not quantities.  A colon/dot between digits is a
            # common OCR substitution for a narrow digit in the game font, so
            # ``3:03`` is retained as the numeric quantity 303.
            raw = record.text.strip().replace(",", "")
            if re.fullmatch(r"\d{1,8}\.?", raw):
                digits = raw.rstrip(".")
            elif re.fullmatch(r"\d{1,2}[:.]\d{1,4}", raw):
                digits = raw.replace(":", "").replace(".", "")
            else:
                continue
            if not digits:
                continue
            row = min(range(2), key=lambda index: abs(centers_y[index] - record.y))
            row_centers = centers_x[row * 4:(row + 1) * 4]
            left = (record.left if record.left is not None else record.x) - 2
            right = (record.right if record.right is not None else record.x + len(digits) * 10.0) + 2
            columns = [
                index for index, center in enumerate(centers_x)
                if left <= center <= right and row * 4 <= index < (row + 1) * 4
            ]
            if not columns:
                columns = [
                    row * 4 + min(range(4), key=lambda index: abs(row_centers[index] - record.x))
                ]
            if len(columns) == 1:
                pieces = [digits]
            else:
                boundaries = [
                    (centers_x[index] + centers_x[index + 1]) / 2
                    for index in columns[:-1]
                ]
                pieces = []
                start = 0
                for boundary in boundaries:
                    fraction = max(0.0, min(1.0, (boundary - left) / max(1.0, right - left)))
                    stop = max(start + 1, min(len(digits) - (len(columns) - len(pieces) - 1), round(len(digits) * fraction)))
                    pieces.append(digits[start:stop])
                    start = stop
                pieces.append(digits[start:])
            for column, piece in zip(columns, pieces):
                if not piece or int(piece) < 0:
                    continue
                slot = str(column + 1)
                value = int(piece)
                previous = counts.get(slot)
                if previous is None or len(str(value)) > len(str(previous)):
                    counts[slot] = value
        return counts

    def _recognize(self, image: Image.Image, *, use_det: bool) -> list[OcrLine]:
        import numpy as np

        raw = self._engine(np.array(image), use_det=use_det, use_cls=False)
        result = raw[0] if isinstance(raw, tuple) else raw
        if not result:
            return []
        records: list[OcrLine] = []
        for item in _as_iterable(result):
            text, box, score = _unpack_record(item)
            if not text:
                continue
            records.append(OcrLine(
                text=text,
                y=_center_y(box),
                x=_center_x(box),
                left=_edge_x(box, min),
                right=_edge_x(box, max),
                score=score,
            ))
        return sorted(records, key=lambda record: record.y if record.y is not None else 0)


def _looks_like_blue_shortcut_cell(image: Image.Image) -> bool:
    """Return whether a shortcut crop contains the blue potion artwork."""
    rgb = image.convert("RGB")
    pixels = list(rgb.getdata())
    if not pixels:
        return False
    blue_pixels = sum(
        1
        for red, green, blue in pixels
        if blue > red + 18 and blue > green + 8
    )
    # Dark UI chrome has a mild blue cast too; the actual MP potion artwork is
    # substantially denser than that background at the crop scale.
    return blue_pixels / len(pixels) >= 0.23


def _blue_shortcut_variant(
    image: Image.Image,
    box: tuple[int, int, int, int],
    scale_x: float,
    scale_y: float,
    *,
    left_pad: int,
    right_pad: int,
) -> Image.Image | None:
    """Build a clean numeric view for a blue potion quantity.

    The blue icon can intrude into the right side of the quantity strip.  The
    blue channel keeps the white outlined digits visible against both the cell
    background and the potion artwork.  Include a little extra width on both
    sides because the neighboring cell's trailing digit is sometimes merged
    into the OCR run; the caller takes the final separated number.  This
    fallback is only used for a blue cell, so the proven HP/ordinary-slot path
    stays unchanged.
    """
    parent_w, parent_h = image.size
    left = max(0, round((box[0] - SHORTCUT_BOX[0] + left_pad) * scale_x))
    top = max(0, round((box[1] - SHORTCUT_BOX[1] + 10) * scale_y))
    right = min(parent_w, round((box[2] - SHORTCUT_BOX[0] + right_pad) * scale_x))
    bottom = min(parent_h, round((box[3] - SHORTCUT_BOX[1] + 2) * scale_y))
    if right <= left or bottom <= top:
        return None

    source = image.crop((left, top, right, bottom)).convert("RGB")
    # The MP icon and its blue highlight are exactly the pixels that confuse
    # the grayscale model.  In the blue channel the white number outline has
    # strong contrast while the colored artwork remains much closer to the
    # dark panel value.
    source = source.getchannel("B")
    resampling = getattr(Image, "Resampling", Image)
    return source.resize(
        (max(1, source.width * 5), max(1, source.height * 5)),
        resampling.LANCZOS,
    )


def _read_blue_shortcut_count(
    ocr: StatPanelOcr,
    image: Image.Image,
    box: tuple[int, int, int, int],
    scale_x: float,
    scale_y: float,
) -> int | None:
    """Read an MP quantity while separating it from its neighbor cell.

    Read the complete isolated cell and several progressively tighter left
    edges.  A suffix such as ``6`` is never preferred over a complete ``86``
    candidate from the same cell.  The views do not extend into the next cell,
    so this recovery path cannot manufacture a value by joining neighbours.
    """
    candidates: list[int] = []

    # The old implementation started with a five-pixel left inset.  At the
    # user's current scale that can remove the leading ``8`` from ``86`` and
    # leave a perfectly plausible-looking ``6``.  Read the complete cell first
    # and then use progressively tighter views only as corroboration.  These
    # views never extend into the neighbouring cell, so preferring the longest
    # numeric candidate cannot turn the adjacent quantity into this slot.
    for left_pad in (0, 2, 4, 5):
        view = _blue_shortcut_variant(
            image, box, scale_x, scale_y, left_pad=left_pad, right_pad=0
        )
        if view is None:
            continue
        text, _records = ocr._read_once(view)
        for match in re.findall(r"\d[\d,]*", text):
            try:
                value = int(match.replace(",", ""))
            except ValueError:
                continue
            if 0 <= value <= 999_999:
                candidates.append(value)

    if not candidates:
        return None

    # A suffix read (6) is less informative than a complete read (86).  The
    # quantity is right-aligned inside this isolated cell, so the greatest
    # digit width is the safest tie-breaker for this colour-specific path.
    return max(candidates, key=lambda value: (len(str(value)), value))


def _as_iterable(value: Any) -> Iterable[Any]:
    if isinstance(value, (list, tuple)):
        return value
    return (value,)


def _unpack_record(item: Any) -> tuple[str, Any, float | None]:
    """Normalize RapidOCR 1.x result variants.

    Some releases return ``[box, text, score]`` while recognition-only mode
    in other releases returns a text-first tuple.  The old wrapper assumed
    ``result[0][0]`` was text; with newer RapidOCR it is the polygon, which
    silently fed lists into the parser.  Detect by type instead of pinning
    the app to one transitive API shape.
    """
    if isinstance(item, dict):
        text = item.get("text") or item.get("txt") or ""
        box = item.get("box") or item.get("points")
        score = item.get("score") or item.get("confidence")
        return str(text), box, _as_score(score)
    if isinstance(item, (list, tuple)):
        if len(item) >= 2 and isinstance(item[1], str):
            return item[1], item[0], _as_score(item[2] if len(item) > 2 else None)
        if item and isinstance(item[0], str):
            return item[0], None, _as_score(item[1] if len(item) > 1 else None)
    return "", None, None


def _as_score(value: Any) -> float | None:
    return float(value) if isinstance(value, Real) else None


def _confidence(records: list[OcrLine]) -> float | None:
    scores = [record.score for record in records if record.score is not None]
    return sum(scores) / len(scores) if scores else None


def _field_is_valid(field: str, text: str) -> bool:
    """Check only the fixed shape of a status field, not its value."""
    if field.upper() in {"HP", "MP"}:
        return bool(re.search(rf"{field}\D{{0,3}}\d+\D+\d+", text, re.IGNORECASE))
    if field.upper() == "LV":
        return bool(re.search(r"LV\.?\D{0,20}\d{1,3}", text, re.IGNORECASE))
    if field.upper() == "EXP":
        return _looks_like_exp(text)
    return bool(text.strip())


def _field_signature(field: str, text: str) -> object:
    """Return a comparable parsed shape for retry consensus."""
    match = re.search(
        rf"{field}\D{{0,3}}(\d+)\D+(\d+)", text, re.IGNORECASE
    ) if field.upper() in {"HP", "MP"} else None
    if match:
        return int(match.group(1)), int(match.group(2))
    if field.upper() == "LV":
        match = re.search(r"LV\.?\D{0,20}(\d{1,3})", text, re.IGNORECASE)
        return int(match.group(1)) if match else text
    if field.upper() == "EXP":
        match = re.search(
            r"EXP\D{0,3}([\d,，]+).*?(\d{3,4}|\d{1,2}[.\s:]\d{2})\s*%",
            text,
            re.IGNORECASE,
        )
        return (int(match.group(1).replace(",", "").replace("，", "")), match.group(2)) if match else text
    return text


_EXP_SHAPE_RE = re.compile(r"EXP\D{0,3}[\d,，]+[\[({].*%", re.IGNORECASE)


def _looks_like_exp(text: str) -> bool:
    return bool(_EXP_SHAPE_RE.search(text))


def _looks_like_exp_but_is_malformed(text: str) -> bool:
    return "EXP" in text.upper() and not _looks_like_exp(text)


def _center_y(box: Any) -> float | None:
    if not isinstance(box, (list, tuple)):
        return None
    points: list[float] = []

    def collect(value: Any) -> None:
        if isinstance(value, (list, tuple)):
            for child in value:
                collect(child)
        elif isinstance(value, (int, float)):
            points.append(float(value))

    collect(box)
    if len(points) < 2:
        return None
    # Polygon coordinates are flattened x,y pairs in all RapidOCR variants.
    ys = points[1::2]
    return sum(ys) / len(ys) if ys else None


def _center_x(box: Any) -> float | None:
    if not isinstance(box, (list, tuple)):
        return None
    points: list[float] = []

    def collect(value: Any) -> None:
        if isinstance(value, (list, tuple)):
            for child in value:
                collect(child)
        elif isinstance(value, (int, float)):
            points.append(float(value))

    collect(box)
    xs = points[0::2]
    return sum(xs) / len(xs) if xs else None


def _edge_x(box: Any, reducer) -> float | None:
    if not isinstance(box, (list, tuple)):
        return None
    points: list[float] = []

    def collect(value: Any) -> None:
        if isinstance(value, (list, tuple)):
            for child in value:
                collect(child)
        elif isinstance(value, (int, float)):
            points.append(float(value))

    collect(box)
    xs = points[0::2]
    return reducer(xs) if xs else None
