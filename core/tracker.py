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
    pixel_landmarks: np.ndarray  # 정수 픽셀 (그리기·기존 코드용)
    # MediaPipe world landmarks: 손 중심을 원점으로 한 **미터 단위** 3D 좌표 (21, 3).
    # normalized_landmarks의 z는 상대 깊이 근사일 뿐이지만, 이쪽은 실제 metric 손 모델이라
    # solvePnP로 손의 3D 위치를 복원하는 데 쓸 수 있다 (core.contact3d). 미제공 시 None.
    world_landmarks: np.ndarray | None = None
    #: 서브픽셀 정밀도 픽셀 좌표 (21, 2). solvePnP처럼 정밀도가 정확도로 직결되는 계산에
    #: 쓴다 — ``pixel_landmarks``의 정수 반올림은 3D 높이에 수십 분의 1 mm 오차를 만든다.
    pixel_landmarks_f: np.ndarray | None = None

    @property
    def index_fingertip(self) -> tuple[int, int]:
        """Return the index fingertip (landmark 8) in pixel coordinates."""
        x, y = self.pixel_landmarks[8]
        return int(x), int(y)

    @property
    def world_landmarks_mm(self) -> np.ndarray | None:
        """World landmarks를 밀리미터 단위로 변환해 반환 (없으면 None)."""
        if self.world_landmarks is None:
            return None
        return self.world_landmarks * 1000.0


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
        world_landmarks = getattr(result, "multi_hand_world_landmarks", None) or []
        tracked: list[TrackedHand] = []
        for index, hand_landmarks in enumerate(result.multi_hand_landmarks):
            normalized = np.asarray(
                [(point.x, point.y, point.z) for point in hand_landmarks.landmark],
                dtype=np.float32,
            )
            pixels_f = np.column_stack(
                (
                    np.clip(normalized[:, 0] * width, 0, width - 1),
                    np.clip(normalized[:, 1] * height, 0, height - 1),
                )
            ).astype(np.float64)
            pixels = pixels_f.astype(np.int32)

            label, score = "Unknown", 0.0
            if index < len(handedness) and handedness[index].classification:
                classification = handedness[index].classification[0]
                label, score = classification.label, float(classification.score)

            world = None
            if index < len(world_landmarks):
                world = np.asarray(
                    [(p.x, p.y, p.z) for p in world_landmarks[index].landmark],
                    dtype=np.float32,
                )

            tracked.append(TrackedHand(label, score, normalized, pixels, world, pixels_f))

        return tracked

    def close(self) -> None:
        self._hands.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

