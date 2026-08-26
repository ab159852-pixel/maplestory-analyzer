"""Lightweight, recognition-only OCR for the fixed MapleStory HUD numbers.

The status panel is a particularly good fit for a recognition-only model:
capture.py already provides one tightly-cropped image per field, so running a
general text detector only adds failure modes around the game's borders and
coloured bars.  The bundled English/numeric PP-OCRv4 model is converted to
ONNX and runs through the application's existing ONNX Runtime dependency.

Chinese context text continues to use the existing RapidOCR path.  Keeping
the two paths separate is intentional: the model specialized for Latin
digits should not be asked to recognize map or job names.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from PIL import Image


MODEL_NAME = "en_PP-OCRv4_mobile_rec"
MODEL_RELATIVE_DIR = Path("paddle_models") / MODEL_NAME

# This is the character dictionary shipped in the model's inference.yml.
# CTC index 0 is the blank token; the space character is appended last.
CHARACTER_DICT = (
    "0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
    "abcdefghijklmnopqrstuvwxyz{|}~!\"#$%&'()*+,-./"
)
CHARACTERS = ("",) + tuple(CHARACTER_DICT) + (" ", " ")


def crop_level_badge(image: Image.Image) -> Image.Image:
    """Return the orange LV-number badge when it can be located safely.

    The level crop also contains the white ``LV.`` label.  On adjacent equal
    digits (for example ``44``), the recognition model can confuse the label
    and collapse the first digit.  The badge has a stable orange background,
    so locating that small colour region gives the recognizer only the level
    digits without adding another screen capture or detector pass.

    If the colour marker is absent (a different theme, a transition frame, or
    an unexpected crop), the original image is returned unchanged.
    """
    import numpy as np

    rgb = np.asarray(image.convert("RGB"))
    if rgb.ndim != 3 or rgb.shape[0] < 3 or rgb.shape[1] < 3:
        return image

    red = rgb[:, :, 0].astype("int16")
    green = rgb[:, :, 1].astype("int16")
    blue = rgb[:, :, 2].astype("int16")
    orange = (
        (red > 120)
        & (green > 40)
        & (blue < 150)
        & (red > green * 1.25)
        & (green > blue * 1.05)
    )
    ys, xs = np.where(orange)
    if not len(xs):
        return image

    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    if right - left < 3 or bottom - top < 3:
        return image

    pad_x = max(1, round(image.width * 0.03))
    pad_y = max(1, round(image.height * 0.10))
    return image.crop(
        (
            max(0, left - pad_x),
            max(0, top - pad_y),
            min(image.width, right + pad_x),
            min(image.height, bottom + pad_y),
        )
    )


def _bundled_model_dir() -> Path | None:
    """Locate the packaged ONNX model in source and PyInstaller layouts."""
    candidates: list[Path] = []
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        root = Path(frozen_root)
        candidates.extend((root / MODEL_RELATIVE_DIR, root / "_internal" / MODEL_RELATIVE_DIR))
    if getattr(sys, "frozen", False):
        root = Path(sys.executable).resolve().parent
        candidates.extend((root / MODEL_RELATIVE_DIR, root / "_internal" / MODEL_RELATIVE_DIR))

    source_root = Path(__file__).resolve().parents[2]
    candidates.append(source_root / "assets" / MODEL_RELATIVE_DIR)

    for candidate in candidates:
        if (candidate / "inference.onnx").is_file():
            return candidate
    return None


def _preprocess(image: Image.Image) -> Any:
    """Match PP-OCR's aspect-preserving 48px resize and right padding."""
    import cv2
    import numpy as np

    bgr = np.asarray(image.convert("RGB"))[:, :, ::-1]
    height, width = bgr.shape[:2]
    resized_width = min(320, max(1, int(np.ceil(48 * width / float(height)))))
    resized = cv2.resize(
        bgr,
        (resized_width, 48),
        interpolation=cv2.INTER_LINEAR,
    )
    # RecResizeImg does not stretch a narrow crop to the full model width;
    # it pads the unused right side.  Keeping this geometry is important for
    # narrow glyphs such as 6/8 and adjacent equal digits such as 44.
    padded = np.zeros((48, 320, 3), dtype=np.uint8)
    padded[:, :resized_width, :] = resized
    normalized = padded.astype("float32").transpose((2, 0, 1)) / 255.0
    normalized = (normalized - 0.5) / 0.5
    return normalized


def _decode(output: Any) -> tuple[str, float]:
    """Greedy CTC decode with the model's bundled character dictionary."""
    import numpy as np

    values = np.asarray(output)
    if values.ndim == 3:
        values = values[0]
    indices = values.argmax(axis=-1)
    probabilities = values.max(axis=-1)
    result: list[str] = []
    selected_scores: list[float] = []
    previous = 0
    for index, probability in zip(indices, probabilities):
        index = int(index)
        if index != previous and index != 0 and index < len(CHARACTERS):
            result.append(CHARACTERS[index])
            selected_scores.append(float(probability))
        previous = index
    return "".join(result), (
        sum(selected_scores) / len(selected_scores) if selected_scores else 0.0
    )


class OnnxNumericRecognizer:
    """Batch-capable numeric recognizer backed by ONNX Runtime."""

    def __init__(self) -> None:
        model_dir = _bundled_model_dir()
        if model_dir is None:
            raise FileNotFoundError(
                f"Bundled {MODEL_NAME} ONNX model was not found"
            )
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            str(model_dir / "inference.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

    def read_fields(self, images: dict[str, Image.Image]) -> dict[str, str]:
        """Recognize all fixed numeric fields in one inference batch."""
        if not images:
            return {}
        import numpy as np

        names = tuple(images)
        batch = np.stack([_preprocess(images[name]) for name in names], axis=0)
        outputs = self._session.run(None, {self._input_name: batch})
        predictions = outputs[0]
        recognized: dict[str, str] = {}
        for index, name in enumerate(names):
            text, _score = _decode(predictions[index])
            if text:
                recognized[name] = text
        return recognized
