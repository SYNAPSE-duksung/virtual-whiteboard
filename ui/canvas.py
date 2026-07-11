""" 스트로크 기록 및 렌더링 엔진

특정 UI 프레임워크에 의존하지 않는 순수 NumPy/OpenCV 기반으로 구현되어,
 OpenCV 데모와 PyQt 애플리케이션에서 모두 공유하여 사용할 수 있습니다. 
 펜의 접촉 및 분리(pen-up/down) 여부는 호출부에서 결정하며, 이 모듈은 오직 기록과 그리기 기능(렌더링)만 수행합니다.
"""

from __future__ import annotations

import cv2
import numpy as np

Point = tuple[int, int]


class StrokeCanvas:
    """Accumulates pen strokes and renders them incrementally."""

    def __init__(
        self,
        width: int,
        height: int,
        *,
        line_color: tuple[int, int, int] = (0, 0, 255),
        line_thickness: int = 4,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        self.width = width
        self.height = height
        self.line_color = line_color
        self.line_thickness = line_thickness
        self.strokes: list[list[Point]] = []
        self._active_stroke: list[Point] | None = None
        # White-background canvas: OCR input for the AI pipeline later on.
        self._image = np.full((height, width, 3), 255, dtype=np.uint8)
        # Ink mask makes overlay() O(pixels) instead of re-drawing every stroke.
        self._mask = np.zeros((height, width), dtype=np.uint8)

    @property
    def is_pen_down(self) -> bool:
        return self._active_stroke is not None

    @property
    def image(self) -> np.ndarray:
        """White-background canvas holding only the ink."""
        return self._image

    def pen_down(self, point: Point) -> None:
        """Start a new stroke at ``point``; ends any stroke in progress."""
        self.pen_up()
        self._active_stroke = [point]

    def move(self, point: Point) -> None:
        """Extend the active stroke to ``point``; ignored while pen is up."""
        if self._active_stroke is None:
            return
        self._draw_segment(self._active_stroke[-1], point)
        self._active_stroke.append(point)

    def pen_up(self) -> None:
        """Finish the active stroke; single-point strokes are discarded."""
        if self._active_stroke is None:
            return
        if len(self._active_stroke) > 1:
            self.strokes.append(self._active_stroke)
        self._active_stroke = None

    def clear(self) -> None:
        """Erase every stroke and reset the canvas."""
        self.strokes.clear()
        self._active_stroke = None
        self._image[:] = 255
        self._mask[:] = 0

    def overlay(self, frame: np.ndarray) -> np.ndarray:
        """Return a copy of ``frame`` with the ink composited on top."""
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"frame size {frame.shape[1]}x{frame.shape[0]} does not match "
                f"canvas size {self.width}x{self.height}"
            )
        composed = frame.copy()
        composed[self._mask > 0] = self.line_color
        return composed

    def _draw_segment(self, start: Point, end: Point) -> None:
        cv2.line(self._mask, start, end, 255, self.line_thickness, cv2.LINE_AA)
        cv2.line(self._image, start, end, self.line_color, self.line_thickness, cv2.LINE_AA)
