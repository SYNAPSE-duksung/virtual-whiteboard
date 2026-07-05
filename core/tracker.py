"""MediaPipe-based hand landmark extraction."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np


@dataclass(frozen=True, slots=True)
class TrackedHand:
    """Landmarks detected for one hand in image and normalized coordinates."""

    handedness: str
    score: float
    normalized_landmarks: np.ndarray
    pixel_landmarks: np.ndarray

    @property
    def index_fingertip(self) -> tuple[int, int]:
        """Return the index fingertip (landmark 8) in pixel coordinates."""
        x, y = self.pixel_landmarks[8]
        return int(x), int(y)


class HandTracker:
    """Small lifecycle-safe wrapper around MediaPipe Hands."""

    def __init__(
        self,
        *,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
    ) -> None:
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, bgr_frame: np.ndarray) -> list[TrackedHand]:
        """Detect hands in a BGR frame; tracking loss returns an empty list."""
        if bgr_frame is None or bgr_frame.size == 0:
            raise ValueError("bgr_frame must be a non-empty image")

        height, width = bgr_frame.shape[:2]
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        result = self._hands.process(rgb_frame)

        if not result.multi_hand_landmarks:
            return []

        handedness = result.multi_handedness or []
        tracked: list[TrackedHand] = []
        for index, hand_landmarks in enumerate(result.multi_hand_landmarks):
            normalized = np.asarray(
                [(point.x, point.y, point.z) for point in hand_landmarks.landmark],
                dtype=np.float32,
            )
            pixels = np.column_stack(
                (
                    np.clip(normalized[:, 0] * width, 0, width - 1),
                    np.clip(normalized[:, 1] * height, 0, height - 1),
                )
            ).astype(np.int32)

            label, score = "Unknown", 0.0
            if index < len(handedness) and handedness[index].classification:
                classification = handedness[index].classification[0]
                label, score = classification.label, float(classification.score)

            tracked.append(TrackedHand(label, score, normalized, pixels))

        return tracked

    def close(self) -> None:
        self._hands.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

