"""OCR adapters for the fixed MapleStory HUD regions.

See VERSIONS.md for why rapidocr-onnxruntime stands in for the spec's originally
named PP-OCRv6-tiny ONNX model -- same OCR family, bundled models, no manual
model-file wiring.

Numeric status fields now prefer the bundled PP-OCR English/numeric
recognition-only model through ONNX Runtime. RapidOCR remains available for
Chinese context regions and as a compatibility fallback when the numeric
model is unavailable.

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
import time
from collections import Counter
from typing import Any, Iterable, Mapping

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from .numeric_ocr import OnnxNumericRecognizer, crop_level_badge
from .regions import (
    MAX_SHORTCUT_QUANTITY,
    Box,
    SHORTCUT_BOX,
    SHORTCUT_SLOT_BOXES,
    shortcut_slot_boxes_for_parent,
)


# Whole-bar detection is useful as a geometry-aware verification pass, but it
# is much more expensive than reading the already-cropped configured cells.
# Run it at the start of a session, after a fast value changes, and occasionally
# as a quiet health check.  This keeps the 0.2s potion worker responsive without
# allowing a clipped fast crop to become the permanent baseline.  The fast
# per-cell path remains active between validations; a failed full detector must
# not force another 2–3 second detector pass on every potion sample.
SHORTCUT_FULL_VALIDATION_INTERVAL_SECONDS = 30.0
# RapidOCR owns separate detector/classifier/recognizer ONNX sessions.  The
# default package configuration uses all CPU cores in each session, which is
# disproportionate for the small fixed crops used here and causes input lag
# when the game is not the foreground window.
RAPIDOCR_INTRA_OP_THREADS = 1
RAPIDOCR_INTER_OP_THREADS = 1


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
        # This model is deliberately recognition-only and is run on the four
        # already-cropped numeric fields as one batch.  Keep a safe RapidOCR
        # fallback for older portable builds that do not contain the model.
        try:
            self._numeric_engine = OnnxNumericRecognizer()
        except Exception:
            self._numeric_engine = None

        # Keep the heavyweight ONNX/RapidOCR import out of module import time.
        # The HUD can paint immediately, then OverlayApp constructs this class
        # on its background loader once the user already has a responsive
        # window.
        from rapidocr_onnxruntime import RapidOCR

        try:
            self._engine = RapidOCR(
                intra_op_num_threads=RAPIDOCR_INTRA_OP_THREADS,
                inter_op_num_threads=RAPIDOCR_INTER_OP_THREADS,
            )
        except TypeError:
            # Keep source builds with an older RapidOCR wrapper usable.  The
            # pinned Windows package accepts these limits; this fallback is
            # only for third-party/custom adapters with the old signature.
            self._engine = RapidOCR()
        self._shortcut_last_fast_counts: dict[str, int] | None = None
        self._shortcut_last_full_counts: dict[str, int] = {}
        self._shortcut_last_validation_at = 0.0
        self._shortcut_validation_signature: tuple[tuple[str, ...], tuple[str, ...]] | None = None

    def read_field(self, image: Image.Image) -> str:
        """Recognition-only OCR on a small pre-cropped single-line field crop.

        This remains the cheap compatibility API used by context OCR and test
        doubles.  The live status path calls :meth:`read_fields`, which knows
        which field it is reading and can validate/retry only the fields whose
        first recognition is structurally weak.
        """
        numeric_text = self._read_numeric_field(image)
        if numeric_text:
            # ``read_field`` is intentionally field-agnostic because it is
            # also used for map/job context OCR.  Only accept a numeric-model
            # result when its shape proves that it is one of the status
            # fields; otherwise a random digit from a Chinese crop can leak
            # into the context parser.
            is_pair = bool(
                re.search(r"\b(?:HP|MP)\D*\d+\D+\d+", numeric_text, re.IGNORECASE)
            )
            is_exp = _looks_like_exp(numeric_text)
            is_level = bool(re.search(r"\b(?:L)?V\.?", numeric_text, re.IGNORECASE))
            if is_level:
                # The full crop can make adjacent equal digits collapse.  A
                # colour-located badge removes the label and recovers the
                # complete value without invoking the slower detector OCR.
                badge_text = self._read_numeric_field(crop_level_badge(image))
                level = _single_level_number(badge_text)
                if level is not None:
                    return f"LV.{level}"
            if is_pair or is_exp:
                repaired = _repair_numeric_result(numeric_text)
                if repaired is not None:
                    return repaired
            # The compatibility API does not receive a field name.  When the
            # numeric result itself proves contradictory (or the LV label was
            # clipped and repeated digits may have collapsed), let RapidOCR
            # re-read this one crop rather than returning a bad value.
            fallback_text, _fallback_records = self._read_once(image)
            if fallback_text:
                return fallback_text
            return numeric_text

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
        numeric_texts: dict[str, str] = {}
        if self._numeric_engine is not None:
            try:
                numeric_images = dict(images)
                if "LV" in numeric_images:
                    # Keep LV in the same one-batch inference as HP/MP/EXP,
                    # but feed the model the colour-located number badge.
                    numeric_images["LV"] = crop_level_badge(numeric_images["LV"])
                numeric_texts = self._numeric_engine.read_fields(numeric_images)
            except Exception:
                # A broken numeric model must not stop the live HUD; switch
                # this instance to the tested RapidOCR fallback.
                self._numeric_engine = None

        result: dict[str, str] = {}
        for name, image in images.items():
            numeric_text = numeric_texts.get(name, "")
            if name.upper() == "LV" and numeric_text:
                # The batch image above is the orange badge, so normalize its
                # single numeric token to the parser's explicit LV shape.
                level = _single_level_number(numeric_text)
                if level is not None:
                    numeric_text = f"LV.{level}"
            level_needs_check = (
                name.upper() == "LV" and _numeric_level_needs_verification(numeric_text)
            )
            if (
                _numeric_text_is_usable(name, numeric_text)
                and _numeric_value_is_plausible(name, numeric_text)
                and not level_needs_check
            ):
                # The recognition model occasionally omits the final percent
                # glyph when it is rendered at the edge of the crop.  The
                # bracketed decimal is still structurally unambiguous, so
                # preserve the fast model result instead of invoking a second
                # engine just to recover the decoration.
                if name.upper() == "EXP" and "%" not in numeric_text:
                    numeric_text = f"{numeric_text}%"
                result[name] = numeric_text
            else:
                result[name] = self._read_typed_field(name, image)
        return result

    def _read_numeric_field(self, image: Image.Image) -> str:
        numeric_engine = getattr(self, "_numeric_engine", None)
        if numeric_engine is None:
            return ""
        try:
            return numeric_engine.read_fields({"field": image}).get("field", "")
        except Exception:
            self._numeric_engine = None
            return ""

    def _read_shortcut_numeric_field(self, image: Image.Image) -> str:
        """Read one isolated shortcut quantity through the digit-only path."""
        numeric_engine = getattr(self, "_numeric_engine", None)
        if numeric_engine is None:
            return ""
        reader = getattr(numeric_engine, "read_digit_fields", None)
        try:
            if callable(reader):
                return reader({"field": image}, max_digits=4).get("field", "")
            # Compatibility for older portable builds and test doubles.
            return numeric_engine.read_fields({"field": image}).get("field", "")
        except Exception:
            self._numeric_engine = None
            return ""

    def reset_shortcut_cache(self) -> None:
        """Forget shortcut pixels when a new session or slot mapping starts.

        The quantity strip intentionally excludes the item artwork, so a
        user can replace the potion in the same key while leaving the number
        unchanged.  Clearing the OCR cache at that boundary prevents the old
        session's quantity from being reused as the new baseline.
        """
        self._shortcut_last_cell_signatures = {}
        self._shortcut_last_cell_values = {}
        self._shortcut_last_fast_counts = {}
        self._shortcut_last_full_counts = {}
        self._shortcut_last_validation_at = 0.0
        self._shortcut_validation_signature = None

    def _read_shortcut_once(self, image: Image.Image) -> tuple[str, list[OcrLine]]:
        """Read a shortcut quantity with a layout-safe numeric fallback.

        Shortcut quantities are the primary source for potion accounting. The
        general RapidOCR recognizer is useful for Chinese UI text, but its
        detector/recognition vocabulary is more likely to return a keyboard
        label or a neighbouring cell. Use the bundled numeric recognizer first
        for an isolated quantity crop and retain RapidOCR only as a
        compatibility fallback for development builds without the numeric
        model.  The previous order let a plausible but wrong RapidOCR value
        such as 1467 overwrite the numeric model's 1487.
        """
        numeric_text = self._read_shortcut_numeric_field(image)
        if _numeric_shortcut_text_is_usable(numeric_text):
            return numeric_text, []
        text, records = self._read_once(image)
        return text, records

    def read_text_field(self, image: Image.Image) -> str:
        """Read a non-numeric UI line with the general RapidOCR model.

        Pickup messages contain the 楓幣 marker as well as digits.  The
        numeric model is intentionally excellent at the latter but cannot
        preserve the Chinese marker needed by the mesos parser, so notification
        rows must explicitly stay on the text model.
        """
        text, _records = self._read_once(image)
        return text

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

    def read_slot_count(
        self,
        image: Image.Image,
        *,
        allow_singleton: bool = False,
        fast: bool = False,
    ) -> int | None:
        """Read a shortcut quantity with numeric-focused OCR consensus.

        Shortcut cells contain both item/count glyphs and keyboard labels such
        as Shift or Ins. A single pass often reads the label or a partial
        digit, so the public/default path accepts a value only when two
        independently preprocessed views agree.  The configured-cell fast
        path may opt into a single clean candidate; temporal baseline and
        economy confirmation still decide whether that candidate is trusted.
        """
        if fast:
            return self._read_slot_count_fast(image, allow_singleton=allow_singleton)

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
        # Do not trust the first integer anymore.  The top part of a shortcut
        # cell contains key labels (Shift/Ins/Ctrl/Del), and a crop that is a
        # few pixels too wide can also expose the neighbouring quantity.  The
        # three views below are intentionally independent: the original keeps
        # the game's glyph geometry, the enlarged view recovers thin strokes,
        # and the lower-right/contrast view suppresses the icon and label.
        candidates: list[int] = []
        for variant in (image, enlarged, lower_right_large, gray):
            text, _records = self._read_shortcut_once(variant)
            value = _extract_shortcut_count(text)
            if value is not None:
                candidates.append(value)
        return _select_slot_consensus(candidates, allow_singleton=allow_singleton)

    def _read_slot_count_fast(
        self,
        image: Image.Image,
        *,
        allow_singleton: bool,
    ) -> int | None:
        """Read a stable isolated cell with two recognition-only views.

        The auxiliary worker runs much more often than the geometry-aware
        full-bar validation. Four OCR calls per cell made an eight-slot
        fallback block for seconds on a busy client. The isolated crop has
        already removed the key label, so the original and one enlarged view
        cover the common path. When those two views disagree, use the same
        lower-right and contrast-normalized recovery views as the public path;
        this fixes thin/outlined digits without paying the extra OCR cost on
        every stable frame.
        """
        resampling = getattr(Image, "Resampling", Image)
        enlarged = image.resize(
            (max(1, image.width * 4), max(1, image.height * 4)),
            resampling.LANCZOS,
        )
        candidates: list[int] = []
        for variant in (image, enlarged):
            text, _records = self._read_shortcut_once(variant)
            value = _extract_shortcut_count(text)
            if value is not None:
                candidates.append(value)
        if len(candidates) >= 2 and len(set(candidates)) > 1:
            lower_right = image.crop((
                image.width // 3,
                image.height // 2,
                image.width,
                image.height,
            ))
            lower_right_large = lower_right.resize(
                (
                    max(1, lower_right.width * 5),
                    max(1, lower_right.height * 5),
                ),
                resampling.LANCZOS,
            )
            gray = ImageOps.autocontrast(ImageOps.grayscale(enlarged))
            gray = ImageEnhance.Contrast(gray).enhance(1.5)
            for variant in (lower_right_large, gray):
                text, _records = self._read_shortcut_once(variant)
                value = _extract_shortcut_count(text)
                if value is not None:
                    candidates.append(value)
        return _select_slot_consensus(candidates, allow_singleton=allow_singleton)

    def read_shortcut_counts(
        self,
        image: Image.Image,
        required_slots: Iterable[str] | None = None,
        blue_slots: Iterable[str] | None = None,
        *,
        allow_full_validation: bool = True,
        slot_images: Mapping[str, Image.Image] | None = None,
    ) -> dict[str, int]:
        """Read shortcut quantities from the requested cells in the parent bar.

        The capture layer first measures the outer frame and then cuts the
        complete eight cells at the actual separator midpoints.  Detection on
        the complete bar remains a legacy diagnostic path only when the caller
        omits ``required_slots``.  Live potion tracking passes the configured
        cell IDs, so an unconfigured cell never reaches OCR.
        """
        has_explicit_required_slots = required_slots is not None
        required = set(required_slots or ())
        blue = set(blue_slots or ())
        if has_explicit_required_slots and not required:
            # ``None`` preserves the legacy full-bar inspection API for manual
            # diagnostics.  An explicitly empty iterable comes from Settings
            # when no potion row is enabled and must perform zero OCR work.
            return {}
        validation_signature = (tuple(sorted(required)), tuple(sorted(blue)))
        now = time.monotonic()
        last_full_counts = dict(getattr(self, "_shortcut_last_full_counts", {}) or {})
        # The fast worker is the newest source of a quantity.  The old
        # "prefer full, otherwise fast" selection meant that one verified
        # full-bar value could hide newer fast values for the other configured
        # cells.  That left a slot without a previous value, allowing an
        # upward digit substitution such as 1359 -> 1959 to enter the live
        # stream.  Merge both caches per slot, with the fast sample winning.
        previous_counts = dict(last_full_counts)
        previous_counts.update(
            dict(getattr(self, "_shortcut_last_fast_counts", {}) or {})
        )
        last_validation_at = float(getattr(self, "_shortcut_last_validation_at", 0.0))
        last_signature = getattr(self, "_shortcut_validation_signature", None)
        configuration_changed = last_signature != validation_signature
        if configuration_changed:
            self._shortcut_last_cell_signatures = {}
            self._shortcut_last_cell_values = {}
        elapsed = now - last_validation_at
        full_validation_due = (
            not required
            or (
                allow_full_validation
                and (
                    configuration_changed
                    or last_validation_at <= 0.0
                    or elapsed >= SHORTCUT_FULL_VALIDATION_INTERVAL_SECONDS
                )
            )
        )

        # Read configured cells on every auxiliary sample. These quantities
        # are the primary real-time signal; waiting for a full-bar detector
        # pass here can miss a 0.3-0.7s potion change. The general detector
        # remains available to callers that explicitly request validation.
        if required:
            try:
                fast_counts = self._read_shortcut_slot_counts(
                    image,
                    required,
                    blue,
                    slot_images=slot_images,
                    previous_counts=previous_counts,
                )
            except TypeError:
                # Keep test doubles and older plug-in OCR adapters that still
                # implement the three-argument private helper working.
                fast_counts = self._read_shortcut_slot_counts(image, required, blue)
        else:
            fast_counts = {}
        if required and not full_validation_due:
            # A temporary numeric miss must not trigger the expensive detector
            # on every 0.1–0.2s sample. Keep the last trusted quantity for the
            # missing cell and allow the next fast frame to recover it.
            counts = _merge_fast_shortcut_counts(previous_counts, fast_counts, required)
            result = {slot: counts[slot] for slot in required if slot in counts}
            # Cache the merged/trusted result, not the raw candidate map.  If
            # a frame is rejected (for example 1359 -> 1959), caching the
            # empty raw map would erase 1359 and let the same bad OCR value
            # through on the next frame as if it were a new baseline.
            self._shortcut_last_fast_counts = dict(result)
            # Fast-only callers still need a remembered configuration so a
            # stable next frame can use the per-cell image cache. The full
            # validation timestamp intentionally remains untouched: the next
            # normal validation call must still be able to run its scheduled
            # geometry-aware health check.
            self._shortcut_validation_signature = validation_signature
            return result

        # Validate the complete bar only when the numeric path is incomplete
        # (or when no required slots were supplied). A positioned full-bar run
        # uses the general text model and can produce a plausible *wrong*
        # number, so it must not replace a missing numeric result in a live
        # configured potion slot. Keep the last verified numeric value until a
        # later batch has a clean read; this is safer than turning a transient
        # blank into a cost event.
        numeric_engine_available = getattr(self, "_numeric_engine", None) is not None
        if required and numeric_engine_available:
            counts = dict(fast_counts)
            for slot in required - counts.keys():
                if slot in previous_counts:
                    counts[slot] = previous_counts[slot]
        else:
            counts = self._shortcut_counts_from_records(image, self.read_lines(image))
        for slot, value in fast_counts.items():
            positioned = counts.get(slot)
            counts[slot] = (
                value
                if positioned is None
                else _prefer_shortcut_numeric_value(positioned, value)
            )
        missing = required - counts.keys()
        if missing and last_signature == validation_signature:
            # A redraw can make both detection and the isolated crop blank.
            # Never publish that transient blank over a known quantity.
            for slot in missing:
                if slot in previous_counts:
                    counts[slot] = previous_counts[slot]
        missing = required - counts.keys()
        if missing and not numeric_engine_available:
            # Detection occasionally loses only the small glyph run for one
            # slot. Retry enhanced full-bar images only for those slots, and
            # never on the stable fast path above.
            candidates: dict[str, list[int]] = {slot: [] for slot in missing}
            for variant in self._retry_variants(image):
                retry = self._shortcut_counts_from_records(variant, self.read_lines(variant))
                for slot in missing:
                    if slot in retry:
                        candidates[slot].append(retry[slot])
            for slot, values in candidates.items():
                if values:
                    # Prefer the widest numeric interpretation when a
                    # sharpened view recovers leading digits that the original
                    # missed.
                    counts[slot] = max(values, key=lambda value: (len(str(value)), value))
        # Apply the trusted-value guard on *every* full-validation pass,
        # including the first pass after Settings changes the configured-cell
        # signature.  The old conditional only protected a stable signature;
        # changing the enabled slot set therefore let a same-length digit
        # substitution such as 1351 -> 1951 escape directly into the HUD and
        # potion ledger.
        if previous_counts:
            counts = _merge_stable_shortcut_counts(previous_counts, counts)
        if required:
            # A full-bar recovery can see neighboring/blank shortcut cells.
            # When the caller names configured slots, never leak those other
            # cells into potion accounting (e.g. slot 8's 915 into slot 7).
            result = {slot: counts[slot] for slot in required if slot in counts}
        else:
            result = counts

        # Keep the verified result separately from the latest fast result.
        # Keep the value that survived the merge as the next frame's previous
        # quantity.  Storing only ``fast_counts`` would erase a trusted value
        # whenever the current frame was blank or rejected as an OCR jump.
        self._shortcut_last_fast_counts = dict(result)
        self._shortcut_last_full_counts = dict(counts)
        # The per-cell pixel cache must contain the value that survived the
        # trusted merge, not the raw candidate captured before full-bar
        # validation.  Otherwise a rejected 1951 can be served from the same
        # crop signature on the next fast tick and bypass the selector again.
        cell_values = dict(getattr(self, "_shortcut_last_cell_values", {}) or {})
        for slot in required:
            slot_key = f"{slot}:{slot in blue}"
            if slot in result:
                cell_values[slot_key] = result[slot]
            else:
                cell_values.pop(slot_key, None)
        self._shortcut_last_cell_values = cell_values
        self._shortcut_last_validation_at = now
        self._shortcut_validation_signature = validation_signature
        return result

    def _read_shortcut_slot_counts(
        self,
        image: Image.Image,
        required_slots: Iterable[str],
        blue_slots: Iterable[str] = (),
        *,
        slot_images: Mapping[str, Image.Image] | None = None,
        previous_counts: Mapping[str, int] | None = None,
    ) -> dict[str, int]:
        """Read configured cells without running full-bar text detection.

        The quantity is rendered at the bottom of each cell. Keep each crop
        inside its own measured cell so a neighbouring quantity cannot be
        joined to the configured slot at a different DPI scale.
        """
        parent_w, parent_h = image.size
        ref_w = SHORTCUT_BOX[2] - SHORTCUT_BOX[0]
        ref_h = SHORTCUT_BOX[3] - SHORTCUT_BOX[1]
        scale_x = parent_w / ref_w
        scale_y = parent_h / ref_h
        local_slot_boxes = shortcut_slot_boxes_for_parent(Box(0, 0, parent_w, parent_h))
        blue_slot_ids = {str(slot_id) for slot_id in blue_slots}
        counts: dict[str, int] = {}
        last_signatures = dict(getattr(self, "_shortcut_last_cell_signatures", {}) or {})
        last_values = dict(getattr(self, "_shortcut_last_cell_values", {}) or {})
        current_signatures: dict[str, tuple] = {}
        current_values: dict[str, int | None] = {}
        pending: dict[str, tuple[str, Image.Image]] = {}
        for slot_id in required_slots:
            box = local_slot_boxes.get(str(slot_id))
            if box is None:
                continue
            # The capture layer already returns the complete measured cell.
            # Keep the keyboard label in this first recognition view: on the
            # real client it gives RapidOCR enough glyph height to read the
            # outlined quantity in one pass, while the cell boundary prevents
            # a neighbouring quantity from being joined to this slot.
            direct_crop = (slot_images or {}).get(str(slot_id))
            if direct_crop is not None:
                # GameWindowCapture/StaticImageCapture calculate each box
                # from the same absolute transform and then subtract the
                # parent origin. Preserve that exact rounding. Re-scaling a
                # parent crop independently can shift a one-pixel outlined
                # digit and turn 1180 into 80.
                crop = direct_crop
            else:
                crop = image.crop(box.as_tuple())
            slot_key = f"{slot_id}:{str(slot_id) in blue_slot_ids}"
            signature = _shortcut_crop_signature(crop)
            current_signatures[slot_key] = signature
            if (
                last_signatures.get(slot_key) == signature
                and slot_key in last_values
                and last_values[slot_key] is not None
            ):
                cached_count = last_values[slot_key]
                current_values[slot_key] = cached_count
                counts[str(slot_id)] = cached_count
                continue
            # A failed OCR result is deliberately not reusable cache data.
            # Reusing a cached None here made an unchanged quantity crop stay
            # permanently unreadable after one ambiguous frame, exactly when
            # a multi-bottle drop needed a fresh read.
            pending[slot_key] = (str(slot_id), crop)

        # The numeric recognizer is cheap when used as one ONNX batch, but
        # eight separate calls (and then a RapidOCR call for every changed
        # cell) made the old path fall behind the game's 0.3-0.7s potion
        # animation. Stable, cross-view-validated cells are skipped by the
        # signature cache above; only a changed strip reaches this batch. The
        # batch contains several colour views for that one cell and remains
        # isolated from neighbouring quantities.
        numeric_counts = self._read_shortcut_numeric_batch(
            pending,
            blue_slot_ids=blue_slot_ids,
            previous_counts=previous_counts,
        )
        numeric_engine_available = getattr(self, "_numeric_engine", None) is not None

        for slot_key, (slot_id, crop) in pending.items():
            # Keep the crop local to this measured cell and read only the
            # quantity strip. The numeric batch is authoritative for this
            # number; RapidOCR is only a fallback when the numeric model has
            # no usable output.
            fast_count = numeric_counts.get(slot_id)
            if fast_count is None and not numeric_engine_available:
                fast_count = self._read_shortcut_quantity_strip(
                    crop,
                    blue=slot_key.endswith(":True"),
                    # ``last_values`` is an OCR cache, not a trusted quantity.
                    # The trusted previous quantity is only a guard against a
                    # multi-digit substitution; it does not make a real
                    # one-bottle decrease impossible.
                    previous=(previous_counts or {}).get(slot_id),
                )
            count = fast_count
            if count is None and not numeric_engine_available:
                # A redraw can temporarily erase both views. Use the existing
                # multi-view cell reader only for that miss. Crucially, never
                # replace a valid numeric-strip result with a whole-cell
                # RapidOCR result: that was the direct cause of 1487 becoming
                # 1467/1438 after the geometry fix.
                count = self.read_slot_count(crop, allow_singleton=True, fast=True)
            if count is None and slot_key.endswith(":True") and not numeric_engine_available:
                # The blue icon can cover a leading stroke.  Keep the colour
                # recovery as a targeted fallback for blanks only; applying it
                # to every blue cell was the source of slow and suffix-heavy
                # reads such as 294 -> 2947.
                box = SHORTCUT_SLOT_BOXES.get(slot_id)
                if box is not None:
                    count = _read_blue_shortcut_count(
                        self,
                        image,
                        box,
                        scale_x,
                        scale_y,
                    )
            if count is not None:
                counts[slot_id] = count
                current_values[slot_key] = count
            else:
                # Do not persist a blank result as if it were a successful
                # OCR decision.  The same crop must be retried next tick.
                current_values.pop(slot_key, None)
        self._shortcut_last_cell_signatures = current_signatures
        self._shortcut_last_cell_values = current_values
        return counts

    def _read_shortcut_numeric_batch(
        self,
        pending: Mapping[str, tuple[str, Image.Image]],
        *,
        blue_slot_ids: set[str],
        previous_counts: Mapping[str, int] | None = None,
    ) -> dict[str, int]:
        """Read pending shortcut quantity strips in one numeric-model batch.

        ``en_PP-OCRv4_mobile_rec`` is a recognition model, not a detector. It
        is therefore appropriate only after the cell and quantity strip have
        already been cropped. RGB, grayscale, two white-glyph threshold views,
        and (for blue cells) colour channels handle the outlined font and item
        artwork. A value is returned only when the threshold family agrees
        with an independent colour view; a disagreement is unreadable rather
        than a reason to guess a new inventory quantity.
        """
        numeric_engine = getattr(self, "_numeric_engine", None)
        if numeric_engine is None or not pending:
            return {}

        batch: dict[str, Image.Image] = {}
        owners: dict[str, str] = {}
        for _slot_key, (slot_id, crop) in pending.items():
            strip = _shortcut_quantity_strip(crop)
            if strip.width <= 1 or strip.height <= 1:
                continue
            views = _shortcut_numeric_views(
                strip,
                blue=slot_id in blue_slot_ids,
                layout_hint=(previous_counts or {}).get(slot_id),
            )
            for view_name, view in views:
                key = f"shortcut:{slot_id}:{view_name}"
                batch[key] = view
                owners[key] = slot_id

        if not batch:
            return {}
        try:
            digit_reader = getattr(numeric_engine, "read_digit_fields", None)
            if callable(digit_reader):
                texts = digit_reader(batch, max_digits=4)
            else:
                # Compatibility for older portable builds and test doubles.
                texts = numeric_engine.read_fields(batch)
        except Exception:
            # Keep the instance usable with the RapidOCR fallback. The next
            # call will not attempt the broken ONNX session again.
            self._numeric_engine = None
            return {}

        candidates: dict[str, list[tuple[int, str]]] = {}
        for key, text in texts.items():
            if not _numeric_shortcut_text_is_usable(text):
                continue
            value = _extract_shortcut_count(text)
            if value is None:
                continue
            view_name = key.rsplit(":", 1)[-1]
            if not _numeric_shortcut_text_is_clean(text):
                # A dotted/colon-separated decode can still be a useful
                # corroboration (183.0 -> 1830), but it must not outweigh a
                # clean white-glyph view or create a false conflict with one.
                view_name = f"soft-{view_name}"
            candidates.setdefault(owners.get(key, ""), []).append((value, view_name))

        result: dict[str, int] = {}
        for slot_id, values in candidates.items():
            if not slot_id:
                continue
            selected = _select_shortcut_numeric_views(
                values,
                previous=(previous_counts or {}).get(slot_id),
            )
            if selected is not None:
                result[slot_id] = selected

        # A three-digit quantity can leave the bottom-right frame stroke in
        # the normal strip.  Do not pay for a second crop on healthy cells;
        # only retry unresolved configured cells with a slightly taller view.
        # The recovery path is intentionally colour-only and requires every
        # independent colour view to agree, so it cannot turn a threshold
        # majority into a guessed quantity.
        recovery_pending = {
            slot_id: item
            for slot_id, item in pending.items()
            if slot_id not in result
        }
        if recovery_pending:
            recovery_batch: dict[str, Image.Image] = {}
            recovery_owners: dict[str, str] = {}
            for _slot_key, (slot_id, crop) in recovery_pending.items():
                strip = _shortcut_quantity_recovery_strip(crop)
                if strip.width <= 1 or strip.height <= 1:
                    continue
                for view_name, view in _shortcut_numeric_views(
                    strip,
                    blue=slot_id in blue_slot_ids,
                ):
                    key = f"shortcut-recovery:{slot_id}:{view_name}"
                    recovery_batch[key] = view
                    recovery_owners[key] = slot_id
            if recovery_batch:
                try:
                    digit_reader = getattr(numeric_engine, "read_digit_fields", None)
                    if callable(digit_reader):
                        recovery_texts = digit_reader(
                            recovery_batch,
                            max_digits=4,
                        )
                    else:
                        recovery_texts = numeric_engine.read_fields(recovery_batch)
                except Exception:
                    recovery_texts = {}
                recovery_candidates: dict[str, list[tuple[int, str]]] = {}
                for key, text in recovery_texts.items():
                    if not _numeric_shortcut_text_is_usable(text):
                        continue
                    value = _extract_shortcut_count(text)
                    if value is None:
                        continue
                    view_name = key.rsplit(":", 1)[-1]
                    if not _numeric_shortcut_text_is_clean(text):
                        view_name = f"soft-{view_name}"
                    recovery_candidates.setdefault(
                        recovery_owners.get(key, ""),
                        [],
                    ).append((value, view_name))
                for slot_id, values in recovery_candidates.items():
                    if not slot_id:
                        continue
                    selected = _select_shortcut_colour_recovery(
                        values,
                        previous=(previous_counts or {}).get(slot_id),
                        minimum_votes=3 if slot_id in blue_slot_ids else 2,
                    )
                    if selected is not None:
                        result[slot_id] = selected
        return result

    def _read_shortcut_quantity_strip(
        self,
        image: Image.Image,
        *,
        blue: bool,
        previous: int | None = None,
    ) -> int | None:
        """Read only the quantity glyphs from one already-isolated cell.

        The capture layer supplies a cell crop with the correct DPI transform.
        Its lower half is stable across client sizes, while the keyboard label
        and item artwork above it are not useful for a numeric read.  Blue MP
        artwork can still leak into grayscale OCR, so include red/green channel
        views and let the suffix/prefix resolver choose the complete value.

        ``previous`` is intentionally only a hint.  The economy tracker remains
        the authority for whether a decrease is a real drink; this function is
        allowed to return a new observation so the UI can show that OCR saw it.
        """
        strip = _shortcut_quantity_strip(image)
        if strip.width <= 1 or strip.height <= 1:
            return None

        numeric_views = _shortcut_numeric_views(
            strip,
            blue=blue,
            layout_hint=previous,
        )
        numeric_candidates: list[tuple[int, str]] = []
        numeric_available = getattr(self, "_numeric_engine", None) is not None
        for view_name, variant in numeric_views:
            try:
                numeric_text = self._read_shortcut_numeric_field(variant)
            except Exception:
                numeric_text = ""
            if _numeric_shortcut_text_is_usable(numeric_text):
                value = _extract_shortcut_count(numeric_text)
                if value is not None:
                    if not _numeric_shortcut_text_is_clean(numeric_text):
                        view_name = f"soft-{view_name}"
                    numeric_candidates.append((value, view_name))

        if numeric_candidates:
            # Once a numeric-model result exists, do not let the general text
            # recognizer override it with a plausible keyboard-label read.
            # If the numeric views disagree, return no value and let the
            # temporal/economy layer wait for a clean frame.
            selected = _select_shortcut_numeric_views(
                numeric_candidates,
                previous=previous,
            )
            if selected is not None:
                return selected

        # Once the numeric engine is loaded, a blank numeric frame must remain
        # blank.  Letting general RapidOCR fill it from a keyboard label or an
        # item-artifact number would defeat the cross-view guard above.
        if numeric_available:
            recovery_strip = _shortcut_quantity_recovery_strip(image)
            recovery_candidates: list[tuple[int, str]] = []
            for view_name, variant in _shortcut_numeric_views(
                recovery_strip,
                blue=blue,
            ):
                try:
                    numeric_text = self._read_shortcut_numeric_field(variant)
                except Exception:
                    numeric_text = ""
                if _numeric_shortcut_text_is_usable(numeric_text):
                    value = _extract_shortcut_count(numeric_text)
                    if value is not None:
                        if not _numeric_shortcut_text_is_clean(numeric_text):
                            view_name = f"soft-{view_name}"
                        recovery_candidates.append((value, view_name))
            recovered = _select_shortcut_colour_recovery(
                recovery_candidates,
                previous=previous,
                minimum_votes=3 if blue else 2,
            )
            if recovered is not None:
                return recovered
            return None

        candidates: list[int] = []
        for _view_name, variant in numeric_views:
            try:
                text, _records = self._read_shortcut_once(variant)
            except Exception:
                continue
            value = _extract_shortcut_count(text)
            if value is not None:
                candidates.append(value)

        if not candidates:
            return None
        if blue:
            return _select_blue_shortcut_candidate(candidates, allow_singleton=True)
        # A non-blue cell needs only one clean recognition view.  Its isolated
        # geometry plus the temporal validator protects the accounting path.
        return _select_slot_consensus(candidates, allow_singleton=True)

    def _read_shortcut_text_batch(
        self,
        images: dict[str, Image.Image],
    ) -> dict[str, str]:
        """Read changed shortcut cells without sharing OCR calls concurrently.

        RapidOCR's internal recognition batch shares one maximum aspect ratio;
        on MapleStory's tiny outlined font that changes the result (for
        example, ``1180`` can collapse to ``1``).  The public wrapper's
        detector also finds the quantity more reliably than direct full-cell
        recognition.  Keep calls on one OCR worker: sharing the same ONNX
        detector concurrently produced nondeterministic values such as
        ``80``/``2893`` from a stable ``1180``/``2833`` frame.  The caller is
        already on a background monitor thread, and the image/signature cache
        means stable frames do not invoke this path again. The economy layer
        still requires temporal confirmation before any decrease becomes a
        drink event.
        """
        if not images:
            return {}
        values: dict[str, str] = {}
        for name, image in images.items():
            try:
                text = self._read_shortcut_once(image)[0]
            except Exception:
                # A single redraw or an older OCR wrapper must not discard the
                # other cells from this sample. The caller will use its
                # targeted two-view fallback only for this blank result.
                continue
            if text:
                values[name] = text
        return values

    @staticmethod
    def _shortcut_counts_from_records(
        image: Image.Image,
        records: list[OcrLine],
    ) -> dict[str, int]:
        if not records:
            return {}

        parent_w, parent_h = image.size
        local_slot_boxes = shortcut_slot_boxes_for_parent(Box(0, 0, parent_w, parent_h))
        centers_x = [
            (box.left + box.right) / 2
            for box in local_slot_boxes.values()
        ]
        centers_y = [
            (local_slot_boxes["1"].top + local_slot_boxes["1"].bottom) / 2,
            (local_slot_boxes["5"].top + local_slot_boxes["5"].bottom) / 2,
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
                if not piece:
                    continue
                slot = str(column + 1)
                value = int(piece)
                if not 0 <= value <= MAX_SHORTCUT_QUANTITY:
                    # A detector run can span multiple cells.  Never let a
                    # merged five-or-more-digit run become a quantity for one
                    # slot; the isolated-cell path will provide the safe read.
                    continue
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
    top_offset: int = 10,
    channel: str = "B",
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
    top = max(0, round((box[1] - SHORTCUT_BOX[1] + top_offset) * scale_y))
    right = min(parent_w, round((box[2] - SHORTCUT_BOX[0] + right_pad) * scale_x))
    bottom = min(parent_h, round((box[3] - SHORTCUT_BOX[1] + 2) * scale_y))
    if right <= left or bottom <= top:
        return None

    source = image.crop((left, top, right, bottom)).convert("RGB")
    # The MP icon and its blue highlight are exactly the pixels that confuse
    # the grayscale model.  In the blue channel the white number outline has
    # strong contrast while the colored artwork remains much closer to the
    # dark panel value.
    if channel in {"R", "G", "B"}:
        source = source.getchannel(channel)
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

    Read the complete isolated cell in the cleanest colour channels first. A
    suffix such as ``6`` is never preferred over a complete ``86`` candidate
    from the same cell. The views do not extend into the next cell, so this
    recovery path cannot manufacture a value by joining neighbours.
    """
    candidates: list[int] = []

    # The red/green channels keep the white outlined quantity visible while
    # suppressing the blue potion artwork. The older five-pixel inset could
    # remove the leading digit (815 -> 15), while the blue channel could turn
    # a complete 294 into 2942. Start at the full cell and slightly higher in
    # the quantity strip, where the keyboard label is already excluded.
    for channel in ("R", "G"):
        view = _blue_shortcut_variant(
            image,
            box,
            scale_x,
            scale_y,
            left_pad=0,
            right_pad=0,
            top_offset=7,
            channel=channel,
        )
        if view is None:
            continue
        read_shortcut_once = getattr(ocr, "_read_shortcut_once", None)
        if callable(read_shortcut_once):
            text, _records = read_shortcut_once(view)
        else:
            text, _records = ocr._read_once(view)
        value = _extract_shortcut_count(text)
        if value is not None:
            candidates.append(value)

    selected = _select_blue_shortcut_candidate(candidates)
    if selected is not None:
        return selected

    for left_pad in (0, 2):
        view = _blue_shortcut_variant(
            image,
            box,
            scale_x,
            scale_y,
            left_pad=left_pad,
            right_pad=0,
            top_offset=10,
            channel="B",
        )
        if view is None:
            continue
        read_shortcut_once = getattr(ocr, "_read_shortcut_once", None)
        if callable(read_shortcut_once):
            text, _records = read_shortcut_once(view)
        else:
            text, _records = ocr._read_once(view)
        value = _extract_shortcut_count(text)
        if value is not None:
            candidates.append(value)

    if not candidates:
        return None

    # MP artwork makes one of the channel views blank surprisingly often.  A
    # lone candidate is still safe to pass upward because it is isolated to a
    # configured cell and must pass the same temporal quantity checks before it
    # can create a cost event.
    return _select_blue_shortcut_candidate(candidates, allow_singleton=True)


def _select_slot_consensus(
    candidates: Iterable[int], *, allow_singleton: bool = False
) -> int | None:
    """Choose a shortcut count only after independent OCR views agree.

    A longest-value tie-break is deliberate: a blue quantity can be read as a
    suffix (``6``) in one view and as the complete value (``86``) in another.
    It is safe only after the values have received the same number of votes;
    this prevents a single wide/neighbor-merged read from winning by length.
    """
    values = [
        value for value in candidates
        if isinstance(value, int) and 0 <= value <= MAX_SHORTCUT_QUANTITY
    ]
    if not values:
        return None
    counts = Counter(values)
    best_votes = max(counts.values())
    if best_votes < 2:
        if allow_singleton and len(values) == 1:
            return values[0]
        return None
    winners = [value for value, votes in counts.items() if votes == best_votes]
    return max(winners, key=lambda value: (len(str(value)), value))


def _extract_shortcut_count(text: str) -> int | None:
    """Extract the final bounded integer from a shortcut OCR result."""
    normalized = str.maketrans("０１２３４５６７８９", "0123456789")
    compact = str(text).translate(normalized).replace(" ", "")
    # The numeric model may render a narrow digit separator as a dot/colon;
    # shortcut quantities are integers, so ``118.0`` and ``3:03`` represent
    # 1180 and 303 rather than decimal values.
    compact = re.sub(r"(?<=\d)[.:](?=\d)", "", compact)
    matches = re.findall(r"\d[\d,]*", compact)
    if not matches:
        return None
    try:
        value = int(matches[-1].replace(",", ""))
    except ValueError:
        return None
    return value if 0 <= value <= MAX_SHORTCUT_QUANTITY else None


def _numeric_shortcut_text_is_usable(text: str) -> bool:
    """Accept numeric-model output only when icon strokes are absent."""
    compact = str(text).translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    if not re.fullmatch(r"[\d\s,.:]+", compact):
        return False
    compact = re.sub(r"[\s,.:]+", "", compact)
    return bool(compact) and compact.isdigit() and _extract_shortcut_count(text) is not None


def _numeric_shortcut_text_is_clean(text: str) -> bool:
    """Return whether a numeric decode contains no OCR punctuation noise."""
    compact = str(text).translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    return bool(re.fullmatch(r"[\d\s,]+", compact))


def _shortcut_numeric_views(
    image: Image.Image,
    *,
    blue: bool,
    layout_hint: int | None = None,
) -> list[tuple[str, Image.Image]]:
    """Build small, independent views for the already isolated quantity strip.

    The quantity is white outlined text drawn over an item icon.  The raw and
    grayscale views preserve the original glyph geometry, while the two
    threshold views remove most coloured artwork and make thin white strokes
    repeatable across a redraw.  The red/green views are especially useful for
    blue MP items, but are also cheap enough to retain as a fallback for any
    cell.  Keep the complete measured strip as the live input: the game uses
    proportional glyph widths, so the left edge moves when a quantity contains
    a narrow ``1``.  The previous dynamic bright-pixel crop changed the aspect
    ratio before OCR and made a small outlined ``3`` unstable; it remains
    available as a diagnostic helper, but must not be the primary live view.
    """
    rgb = image.convert("RGB")
    prefix = "raw-"
    gray = ImageOps.autocontrast(ImageOps.grayscale(rgb))
    gray_rgb = ImageEnhance.Contrast(gray).enhance(1.35).convert("RGB")
    views: list[tuple[str, Image.Image]] = [
        (f"{prefix}rgb", rgb),
        (f"{prefix}gray", gray_rgb),
    ]
    for threshold in (170, 180):
        white = gray.point(lambda pixel, threshold=threshold: 255 if pixel >= threshold else 0)
        views.append((f"{prefix}white{threshold}", white.convert("RGB")))
    if blue:
        views.extend(
            (f"{prefix}{channel.lower()}", rgb.getchannel(channel).convert("RGB"))
            for channel in ("R", "G")
        )

    return views


def _right_aligned_quantity_crop(image: Image.Image) -> Image.Image:
    """Crop a proportional quantity by its actual right-anchored glyph run.

    The game does not reserve four equal character cells for a quantity.  A
    narrow leading ``1`` moves the left edge of the whole string while the
    right edge remains at the same place.  This helper deliberately detects
    only the horizontal extent of the bright, low-chroma glyph run and keeps
    the complete vertical strip.  It is a geometry normalizer, not a number
    guesser: if the evidence is too weak or too close to a cell border, the
    original strip is returned unchanged and the existing multi-view guard
    remains responsible for rejecting the frame.

    The mask is intentionally conservative.  The quantity's white outline is
    achromatic, while the item artwork is usually coloured; choosing the
    rightmost plausible cluster also prevents a bright icon fragment on the
    left from becoming part of the number.  A small pad protects the first and
    last anti-aliased strokes without bringing the neighbouring cell into the
    crop.
    """
    width, height = image.size
    if width <= 4 or height <= 3:
        return image

    try:
        import numpy as np

        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[0] != height or rgb.shape[1] != width:
            return image
        # Luma and chroma are enough here; do not run a full connected
        # component detector on every 0.2–0.3s frame.
        luma = (
            rgb[:, :, 0].astype("float32") * 0.299
            + rgb[:, :, 1].astype("float32") * 0.587
            + rgb[:, :, 2].astype("float32") * 0.114
        )
        chroma = rgb.max(axis=2).astype("int16") - rgb.min(axis=2).astype("int16")
        y0 = max(0, round(height * 0.05))
        y1 = min(height, max(y0 + 1, round(height * 0.91)))
        if y1 - y0 < 3:
            return image

        masks = (
            (luma >= 155) & (chroma <= 100),
            (luma >= 185) & (chroma <= 135),
        )
        candidates: list[tuple[float, int, int]] = []
        max_gap = max(2, round(width * 0.08))
        for mask in masks:
            active = mask[y0:y1].sum(axis=0) >= 1
            # The measured cell boundary is not a glyph.  Keep the rightmost
            # columns available: the final outlined digit may touch that edge
            # after DPI scaling, and deleting them is how a complete value
            # becomes 157/57.  Only the left border is excluded as an anchor.
            active[:2] = False
            indices = np.flatnonzero(active)
            if not len(indices):
                continue

            start = int(indices[0])
            previous = start
            runs: list[tuple[int, int]] = []
            for value in indices[1:]:
                current = int(value)
                if current - previous > max_gap:
                    runs.append((start, previous + 1))
                    start = current
                previous = current
            runs.append((start, previous + 1))

            if not runs:
                continue

            # A proportional string is made of several narrow glyph runs.
            # Join nearby runs from the right instead of selecting only the
            # last connected component; the latter loses the leading 1 (or
            # any digit separated by a wider anti-aliased gap).  The gap is
            # relative to the measured cell so this remains DPI-safe.
            right = runs[-1][1]
            left = runs[-1][0]
            join_gap = max(3, round(width * 0.14))
            for previous_left, previous_right in reversed(runs[:-1]):
                if left - previous_right > join_gap:
                    break
                left = previous_left
            run_width = right - left
            if run_width < 3 or run_width > max(6, round(width * 0.98)):
                continue
            # A quantity is rendered toward the right side of its cell.
            # Reject isolated artwork at the far left, but allow a single
            # narrow digit such as ``1``.
            if right < round(width * 0.38):
                continue
            area = int(mask[y0:y1, left:right].sum())
            if area < max(3, round(height * 0.10)):
                continue
            score = float(right) * 10.0 + min(area, 100) * 0.01
            candidates.append((score, left, right))

        if not candidates:
            return image
        _score, left, right = max(candidates)
        left_pad = max(1, round(width * 0.04))
        crop_left = max(0, left - left_pad)
        # Right-align every candidate to the complete cell edge.  Do not
        # crop at the detected bright-pixel edge: anti-aliased strokes and a
        # narrow final digit can occupy the last one or two columns.
        crop_right = width
        if crop_right - crop_left < 3 or crop_right <= crop_left:
            return image
        # Do not normalize a candidate that nearly equals the entire cell;
        # that is usually a border/transition frame, not a useful digit run.
        if crop_right - crop_left >= round(width * 0.97):
            return image
        return image.crop((crop_left, 0, crop_right, height))
    except Exception:
        # Numeric OCR must remain available in minimal source environments and
        # should never fail merely because this optional geometry hint failed.
        return image


def _shortcut_layout_pattern(value: int | None) -> str | None:
    """Describe a four-digit quantity by the glyphs that change its width."""
    if not isinstance(value, int) or not 1000 <= value <= MAX_SHORTCUT_QUANTITY:
        return None
    text = str(value)
    if len(text) != 4:
        return None
    return "".join("1" if digit == "1" else "x" for digit in text)


def _shortcut_pattern_quantity_crop(
    image: Image.Image, layout_hint: int | None
) -> Image.Image | None:
    """Create a right-anchored safety crop for a known digit-width pattern.

    The game does not use four equal character cells. This is deliberately a
    *safety* crop: it may retain a little extra left-side background, but it
    cannot remove the leading glyph. Each occurrence of the narrow ``1``
    expands the left margin, so patterns such as ``11xx`` and ``111x`` get a
    different range without a table of fragile absolute coordinates.
    """
    pattern = _shortcut_layout_pattern(layout_hint)
    if pattern is None:
        return None
    width, height = image.size
    if width <= 4 or height <= 3:
        return None

    detected = _right_aligned_quantity_crop(image)
    # ``_right_aligned_quantity_crop`` returns the original image when the
    # bright glyph mask is temporarily unavailable.  Treat that as "no
    # detected left edge", not as a left edge at zero; otherwise the fallback
    # pattern range would silently become the same full-cell crop for every
    # layout hint.
    detected_left = width if detected is image else max(0, width - detected.width)
    narrow_count = pattern.count("1")
    # Leave a generous base envelope for four ordinary glyphs, then widen it
    # toward the left for every narrow 1.  Subtract the narrow-glyph margin:
    # adding a ``1`` must move the left edge left, never right.  All ratios are
    # based on the actual cell width, so the same module follows DPI/client
    # scaling.
    base_width = max(1, round(width * 0.78))
    narrow_margin = max(1, round(width * 0.04))
    safety_margin = max(1, round(width * 0.05))
    pattern_left = max(
        0,
        width - base_width - narrow_count * narrow_margin - safety_margin,
    )
    # Never move right of the pixel-derived crop: a weak pattern hint must
    # only enlarge the trusted envelope, not introduce a new clipping edge.
    left = min(detected_left, pattern_left)
    if left >= width - 2:
        return None
    return image.crop((left, 0, width, height))


def _select_shortcut_numeric_views(
    candidates: Iterable[tuple[int, str]], *, previous: int | None
) -> int | None:
    """Select a quantity only when colour and white-glyph views corroborate.

    A generic recognizer can consistently turn a game-font ``5`` into ``3``;
    ordinary majority voting would then make the wrong value look reliable.
    Prefer a value that appears in both a threshold view and an independent
    colour view.  If a threshold view conflicts with the colour views, return
    ``None`` and let the previous trusted quantity remain visible.  This is the
    key distinction between an unreadable frame and a real inventory change.
    """
    values = [
        (value, str(view_name))
        for value, view_name in candidates
        if isinstance(value, int) and 0 <= value <= MAX_SHORTCUT_QUANTITY
    ]
    if previous is not None:
        values = [
            (value, view_name)
            for value, view_name in values
            if _shortcut_numeric_transition_is_usable(previous, value)
        ]
    if not values:
        return None

    threshold_values = Counter(
        value
        for value, view_name in values
        if _numeric_view_base_name(view_name).startswith("white")
    )
    colour_values = Counter(
        value
        for value, view_name in values
        if not str(view_name).startswith("soft-")
        and not _numeric_view_base_name(view_name).startswith("white")
    )
    if threshold_values:
        # A threshold candidate is the protected white-glyph interpretation.
        # If more than one threshold value survives, the frame is ambiguous.
        best_threshold_votes = max(threshold_values.values())
        threshold_winners = {
            value for value, votes in threshold_values.items() if votes == best_threshold_votes
        }
        if len(threshold_winners) != 1:
            return None
        threshold_value = next(iter(threshold_winners))
        if colour_values:
            # Require at least one independent colour view to agree.  A
            # conflicting raw result such as 320 beside threshold 3204 is
            # intentionally rejected rather than resolved by vote count.
            if colour_values.get(threshold_value, 0) <= 0:
                return None
            # Do not let a majority of colour preprocessings hide a
            # same-width disagreement.  This is the failure mode behind a
            # real 920 being promoted to 320: three views see 320 while the
            # untouched RGB view sees 920.  Both are syntactically valid, so
            # neither voting nor a magnitude heuristic is safe.  Returning
            # None keeps the previous trusted quantity and avoids a false
            # inventory change/cost event.
            if any(
                value != threshold_value
                and len(str(value)) == len(str(threshold_value))
                for value in colour_values
            ):
                return None
        elif best_threshold_votes < 2:
            return None
        return threshold_value

    # If the threshold views are blank, two independent colour views may still
    # be sufficient during a brief redraw. A singleton is never a baseline.
    best_votes = max(colour_values.values()) if colour_values else 0
    winners = [value for value, votes in colour_values.items() if votes == best_votes]
    if best_votes < 2 or len(winners) != 1:
        return None
    return winners[0]


def _select_shortcut_colour_recovery(
    candidates: Iterable[tuple[int, str]],
    *,
    previous: int | None,
    minimum_votes: int,
) -> int | None:
    """Select a fallback quantity only when all hard colour views agree.

    The fallback crop is used after the normal threshold/colour families
    conflict.  It may recover a three-digit value whose last glyph is followed
    by the frame border, but it must not resolve two equally plausible digit
    strings by majority vote.  Soft punctuation results and white-threshold
    views are deliberately excluded from this recovery family.
    """
    values = [
        (value, str(view_name))
        for value, view_name in candidates
        if isinstance(value, int)
        and 0 <= value <= MAX_SHORTCUT_QUANTITY
        and not str(view_name).startswith("soft-")
        and not _numeric_view_base_name(view_name).startswith("white")
    ]
    if not values:
        return None
    counts = Counter(value for value, _view_name in values)
    if len(counts) != 1:
        return None
    value, votes = next(iter(counts.items()))
    if votes < max(2, int(minimum_votes)):
        return None
    if previous is not None and not _shortcut_numeric_transition_is_usable(
        previous,
        value,
    ):
        return None
    return value


def _numeric_view_base_name(view_name: str) -> str:
    """Remove preprocessing prefixes before classifying a numeric view.

    Dynamic/pattern quantity normalization adds prefixes to the view name.
    The selector still needs to recognize ``aligned-white170`` and
    ``layout-white175`` as protected threshold views while treating their RGB
    partners as independent colour views. Keep this normalization local to
    the selector so existing cache/test names (``white170``, ``soft-rgb``)
    remain valid.
    """
    name = str(view_name)
    if name.startswith("soft-"):
        name = name[5:]
    changed = True
    while changed:
        changed = False
        for prefix in ("aligned-", "layout-", "raw-"):
            if name.startswith(prefix):
                name = name[len(prefix):]
                changed = True
                break
    return name


def _select_shortcut_numeric_candidate(
    candidates: Iterable[int],
    *,
    previous: int | None,
    blue: bool,
) -> int | None:
    """Select numeric-model output without deciding its cost immediately.

    The quantity font is small enough that two different digit shapes can
    both be syntactically valid. Keep clear truncations out of the OCR stream,
    but pass a complete multi-bottle decrease to the economy layer, where two
    frames and later upward correction can distinguish it from an OCR jump.
    """
    values = [
        value
        for value in candidates
        if isinstance(value, int) and 0 <= value <= MAX_SHORTCUT_QUANTITY
    ]
    if not values:
        return None
    if previous is not None:
        plausible = [
            value
            for value in values
            if _shortcut_numeric_transition_is_usable(previous, value)
        ]
        if not plausible:
            return None
        votes = Counter(plausible)
        return max(plausible, key=lambda value: (votes[value], value))
    if blue:
        return _select_blue_shortcut_candidate(values, allow_singleton=True)
    return _select_slot_consensus(values, allow_singleton=True)


def _select_blue_shortcut_candidate(
    candidates: Iterable[int], *, allow_singleton: bool = False
) -> int | None:
    """Prefer complete blue-cell values over OCR prefixes/suffixes.

    Channel changes can produce ``15`` from ``815`` or ``2942`` from ``294``.
    A suffix means the longer candidate recovered a leading digit; a prefix
    means the longer candidate gained an extra trailing stroke. Resolve those
    relationships before ordinary voting, then fall back to the normal
    independent-view consensus.
    """
    values = [
        value for value in candidates
        if isinstance(value, int) and 0 <= value <= MAX_SHORTCUT_QUANTITY
    ]
    if not values:
        return None
    unique = sorted(set(values), key=lambda value: (len(str(value)), value), reverse=True)
    for longer in unique:
        longer_text = str(longer)
        for shorter in unique:
            shorter_text = str(shorter)
            if len(shorter_text) >= len(longer_text):
                continue
            if longer_text.endswith(shorter_text):
                return longer
            if longer_text.startswith(shorter_text):
                return shorter
    return _select_slot_consensus(values, allow_singleton=allow_singleton)


def _shortcut_quantity_strip(image: Image.Image) -> Image.Image:
    """Return the proportional bottom strip that contains the quantity text."""
    width, height = image.size
    if width <= 1 or height <= 1:
        return image.crop((0, 0, max(1, width), max(1, height)))
    # The outlined quantity starts one pixel above the old 48% split on the
    # reference client.  At small cells that single row contains the top bar of
    # the leading digit; losing it is exactly what made 1209/1830 look like a
    # two- or three-digit value.  Remove only the final 5% of the cell: it is
    # the bright separator/shadow line below the number, not part of the last
    # glyph, and it was causing the model to append a spurious digit.
    top = max(0, min(height - 1, round(height * 0.45)))
    bottom = max(top + 1, min(height, round(height * 0.95)))
    # The caller now supplies the complete cell, split at the real separator
    # midpoint.  Do not trim the right edge: the fourth glyph of a 4-digit
    # quantity can occupy those final pixels (the old proportional trim was
    # exactly why values such as ``1570`` could become ``157``/``57``).
    return image.crop((0, top, width, bottom))


def _shortcut_quantity_recovery_strip(image: Image.Image) -> Image.Image:
    """Return a taller retry strip for an unresolved quantity.

    At the reference scale the normal 45%-95% strip is the best four-digit
    envelope.  A three-digit value can nevertheless be classified more
    cleanly when the crop includes the lower frame and starts a little higher;
    this view is therefore only a conflict recovery, never the primary OCR
    input.
    """
    width, height = image.size
    if width <= 1 or height <= 1:
        return image.crop((0, 0, max(1, width), max(1, height)))
    top = max(0, min(height - 1, round(height * 0.40)))
    return image.crop((0, top, width, height))


def _shortcut_crop_signature(image: Image.Image) -> tuple:
    """Return a cheap signature for the quantity strip, not the icon.

    The shortcut icons can animate continuously. Comparing only a reduced
    grayscale bottom strip lets the monitor skip OCR while the quantity stays
    unchanged, while any actual inventory digit change still wakes the cell.
    """
    strip = _shortcut_quantity_strip(image).convert("L")
    reduced = strip.resize(
        (max(8, strip.width // 2), max(6, strip.height // 2)),
        getattr(Image, "Resampling", Image).BILINEAR,
    )
    return reduced.size, hash(reduced.tobytes())


def _is_probable_shortcut_ocr_change(previous: int, current: int) -> bool:
    """Accept a non-truncated quantity transition from a cached value.

    The economy layer performs the temporal/multi-frame decision.  The OCR
    layer must not discard a real multi-bottle decrease or a correction after
    a provisional drop before that layer can see it; its job here is to remove
    clear suffix/truncation reads and the known upward digit-substitution
    pattern.
    """
    if current == previous:
        return True
    if current > previous:
        # Upward values are needed for a correction such as 40 -> 80 after a
        # provisional false drop.  Only block the characteristic same-length
        # substitution/neighbor merge here; EconomyTracker applies the
        # stronger temporal gate before treating any upward value as a real
        # restock or correction.
        return not _is_probable_shortcut_upward_artifact(previous, current)
    if current == 0 and previous <= 10:
        return True
    if _is_shortcut_numeric_truncation(previous, current):
        return False
    return True


def _is_shortcut_numeric_truncation(previous: int, current: int) -> bool:
    """Reject a value that visibly lost one or more leading digits."""
    previous_text = str(previous)
    current_text = str(current)
    if (
        len(current_text) < len(previous_text)
        and bool(current_text)
        and previous_text.endswith(current_text)
    ):
        return True
    # Preserve the narrow one-digit transition used for a real 10 -> 9
    # decrease; larger changes are almost always a clipped leading glyph.
    return previous >= 10 and current < 10 and previous - current > 1


_SHORTCUT_DIGIT_CONFUSIONS = frozenset({
    # Common outlined-font substitutions seen in the bundled numeric model.
    # Keep the list deliberately small: a normal quantity decrease must still
    # be allowed through to the reversible economy layer.
    (3, 9), (9, 3),
    (3, 5), (5, 3),
    (6, 8), (8, 6),
    (0, 8), (8, 0),
})


def _shortcut_digit_substitution(
    previous: int, current: int
) -> tuple[int, int, int, int] | None:
    """Describe a one-position substitution within one shortcut quantity.

    Returning the changed index, old digit, new digit, and decimal place keeps
    the guard tied to the same four-digit slot.  It deliberately does not
    reject a normal decrease; callers decide whether the direction and size
    make the substitution suspicious.
    """
    if not isinstance(previous, int) or not isinstance(current, int):
        return None
    previous_text = str(previous)
    current_text = str(current)
    if len(previous_text) != len(current_text):
        return None
    differences = [
        index
        for index, (old_digit, new_digit)
        in enumerate(zip(previous_text, current_text))
        if old_digit != new_digit
    ]
    if len(differences) != 1:
        return None
    index = differences[0]
    place = 10 ** (len(previous_text) - index - 1)
    return index, int(previous_text[index]), int(current_text[index]), place


def _is_probable_shortcut_upward_artifact(previous: int, current: int) -> bool:
    """Return whether an upward quantity is likely a digit/cell OCR error.

    A real correction after a provisional drop must remain possible, so this
    is intentionally narrower than a blanket ``current > previous`` check.
    The reported ``1359 -> 1959`` case is a same-length, one-digit upward
    substitution (3 -> 9) and is rejected before it reaches accounting.
    Values that add a neighboring cell's suffix/prefix are rejected as well.
    """
    if not isinstance(previous, int) or not isinstance(current, int):
        return False
    if current <= previous:
        return False
    previous_text = str(previous)
    current_text = str(current)

    # A longer value containing the complete old value is the usual symptom
    # of a neighboring cell being joined to this one (86 -> 865, 118 -> 1180).
    if len(current_text) > len(previous_text) and (
        current_text.startswith(previous_text)
        or current_text.endswith(previous_text)
    ):
        return True

    substitution = _shortcut_digit_substitution(previous, current)
    if substitution is None:
        return False

    _index, old_digit, new_digit, place = substitution
    upward_delta = (new_digit - old_digit) * place
    if (old_digit, new_digit) in _SHORTCUT_DIGIT_CONFUSIONS and place >= 10:
        return True
    # A one-character OCR mistake in a high-order position creates a large
    # jump that is not a normal one-bottle pickup. Keep the threshold relative
    # to the actual stack so it also works below 1000.
    return upward_delta >= max(300, int(previous * 0.35))


def _shortcut_numeric_transition_is_usable(previous: int, current: int) -> bool:
    """Allow real decreases/corrections while rejecting obvious OCR jumps."""
    if current <= previous:
        return not _is_shortcut_numeric_truncation(previous, current)
    return not _is_probable_shortcut_upward_artifact(previous, current)


def _merge_fast_shortcut_counts(
    cached: dict[str, int], fast: dict[str, int], required: set[str]
) -> dict[str, int]:
    """Merge low-latency cell reads without publishing OCR jumps."""
    result: dict[str, int] = {}
    for slot in required:
        current = fast.get(slot)
        previous = cached.get(slot)
        if current is None:
            if previous is not None:
                result[slot] = previous
        elif previous is None or _is_probable_shortcut_ocr_change(previous, current):
            result[slot] = current
        elif previous is not None:
            result[slot] = previous
    return result


def _merge_stable_shortcut_counts(
    previous: dict[str, int], current: dict[str, int]
) -> dict[str, int]:
    """Preserve a known value when a full-bar pass returns an OCR jump."""
    result = dict(current)
    for slot, old_value in previous.items():
        new_value = result.get(slot)
        if new_value is None or not _is_probable_shortcut_ocr_change(old_value, new_value):
            result[slot] = old_value
    return result


def _prefer_shortcut_numeric_value(positioned: int, numeric: int) -> int:
    """Prefer numeric-cell OCR unless the full-bar value explains a crop edge.

    The full-bar detector retains x/y geometry but uses the general text model;
    the isolated cell uses the numeric model but can clip one edge.  If one
    value is an obvious prefix/suffix of the other, the positioned run repairs
    that edge.  For unrelated values, keep the numeric model's result because
    it is the source intended for potion accounting.
    """
    positioned_text = str(positioned)
    numeric_text = str(numeric)
    if (
        len(positioned_text) > len(numeric_text)
        and (
            positioned_text.endswith(numeric_text)
            or positioned_text.startswith(numeric_text)
        )
    ):
        return positioned
    if len(numeric_text) > len(positioned_text) and numeric_text.startswith(positioned_text):
        return positioned
    return numeric


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


def _numeric_text_is_usable(field: str, text: str) -> bool:
    """Accept a numeric-model result only when its shape is plausible.

    The LV crop intentionally clips the left side of the label at some
    render scales, so the model may return ``V.69`` rather than ``LV.69``.  The
    parser already has a bounded single-number fallback for that crop; the
    other fields retain their stronger structural requirements.
    """
    if not text.strip():
        return False
    upper = field.upper()
    if upper in {"HP", "MP"}:
        return bool(re.search(r"\d+\D+\d+", text))
    if upper == "EXP":
        return bool(
            re.search(
                r"EXP\D{0,3}\d{3,}\s*[\[({]\s*\d{1,2}[.\s:]\d{2}\s*%?",
                text,
                re.IGNORECASE,
            )
        )
    if upper == "LV":
        # The crop can clip the leading L, leaving ``V.69``.  That is still a
        # valid level result; the one-digit case is checked separately because
        # greedy CTC can collapse adjacent equal digits (``44`` -> ``4``).
        return bool(re.search(r"(?<!\d)\d{1,3}(?!\d)", text))
    return True


def _numeric_value_is_plausible(field: str, text: str) -> bool:
    """Reject physically impossible numeric reads before they enter state."""
    if field.upper() not in {"HP", "MP"}:
        return True
    match = re.search(r"(\d+)\D+(\d+)", text)
    if not match:
        return False
    current, maximum = (int(match.group(index)) for index in (1, 2))
    # A current resource cannot exceed its maximum.  This catches the common
    # one-glyph confusion such as the real 2816 being read as 2818, while the
    # fallback path gets a chance to re-read that single field.
    return maximum > 0 and current <= maximum


def _single_level_number(text: str) -> int | None:
    """Extract one bounded level token from the isolated orange badge."""
    numbers = re.findall(r"(?<!\d)(\d{1,3})(?!\d)", text or "")
    if len(numbers) != 1:
        return None
    return int(numbers[0])


def _numeric_level_needs_verification(text: str) -> bool:
    """Use the compatibility OCR only for an ambiguous one-digit level."""
    match = re.search(r"(?<!\d)(\d{1,3})(?!\d)", text)
    return bool(
        match
        and (len(match.group(1)) == 1 or match.group(1).startswith("0"))
    )


def _repair_numeric_result(text: str) -> str | None:
    """Repair safe decorations or reject an obviously ambiguous OCR result.

    ``read_field`` is retained as a field-agnostic compatibility API, so this
    helper uses only evidence present in the recognized string itself.
    """
    if re.search(r"\bV\.?\s*\.\s*\d{1,3}\b", text, re.IGNORECASE) and not re.search(
        r"\bLV\b", text, re.IGNORECASE
    ):
        # Adjacent equal level digits can collapse in greedy CTC decoding.
        return None
    if re.search(r"\bLV\b", text, re.IGNORECASE) and not re.search(r"\d", text):
        return None
    pair = re.search(r"\b(?:HP|MP)\D*(\d+)\D+(\d+)", text, re.IGNORECASE)
    if pair and int(pair.group(1)) > int(pair.group(2)):
        return None
    if re.search(
        r"\bEXP\D{0,3}\d{3,}\s*[\[({]\s*\d{1,2}[.\s:]\d{2}\s*[\])}]?\s*$",
        text,
        re.IGNORECASE,
    ):
        closing = text[-1:] if text[-1:] in "]]})" else ""
        return f"{text[:-1] if closing else text}%{closing}"
    return text


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
