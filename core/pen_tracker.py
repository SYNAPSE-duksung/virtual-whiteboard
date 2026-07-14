"""고수준 펜 추적기.

``HandTracker``(랜드마크) + ``FingertipSmoother``(스무딩) + ``PenStateDetector``(자동
pen-up/down)를 하나로 조합한다. controller/ui는 이 모듈만 사용하면 되고, 프레임을 넣으면
스무딩된 손끝 좌표·펜 상태·지우기 제스처가 담긴 ``PenFrame``을 돌려받는다.

``tracker.py``의 규약(트래킹 손실 시 예외 대신 빈 결과)을 따른다 — 손을 잃으면
``PenFrame.empty()``를 반환하고 내부 필터/상태를 리셋한다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from core.filters import FingertipSmoother
from core.pen_state import INDEX_TIP, PenStateDetector, compute_pen_ratio, is_open_hand
from core.tracker import HandTracker


@dataclass(frozen=True, slots=True)
class PenFrame:
    """한 프레임의 펜 추적 결과."""

    hand_detected: bool
    fingertip: tuple[int, int] | None  # 스무딩된 픽셀 좌표 = 펜 입력점 (filtered_x/y)
    raw_fingertip: tuple[float, float] | None  # 필터 이전 원본 픽셀 좌표 (float)
    raw_z: float | None  # 검지 끝 정규화 z (깊이 근사, CSV 기록용)
    pen_down: bool
    erase_gesture: bool
    pen_ratio: float | None

    @classmethod
    def empty(cls) -> "PenFrame":
        """트래킹 손실 프레임."""
        return cls(False, None, None, None, False, False, None)


class PenTracker:
    """프레임 in → ``PenFrame`` out. 내부 상태(필터/히스테리시스)를 프레임 간 유지한다."""

    def __init__(
        self,
        *,
        max_num_hands: int = 1,
        min_cutoff: float = 1.0,
        beta: float = 0.3,
        d_cutoff: float = 1.0,
        pen_down_thresh: float = 0.55,
        pen_up_thresh: float = 0.70,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.6,
    ) -> None:
        self._tracker = HandTracker(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._smoother = FingertipSmoother(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self._pen_state = PenStateDetector(down_thresh=pen_down_thresh, up_thresh=pen_up_thresh)

    def process(self, bgr_frame, *, timestamp: float | None = None) -> PenFrame:
        """BGR 프레임을 처리해 ``PenFrame`` 반환.

        ``timestamp``는 One Euro Filter의 시간 기준(초). 미지정 시 ``time.perf_counter()`` 사용.
        """
        hands = self._tracker.process(bgr_frame)
        if not hands:
            self._smoother.reset()
            self._pen_state.reset()
            return PenFrame.empty()

        hand = hands[0]
        height, width = bgr_frame.shape[:2]
        t = time.perf_counter() if timestamp is None else timestamp

        landmarks = hand.normalized_landmarks
        raw_x = float(landmarks[INDEX_TIP][0]) * width
        raw_y = float(landmarks[INDEX_TIP][1]) * height
        raw_z = float(landmarks[INDEX_TIP][2])
        smooth_x, smooth_y = self._smoother.update(t, raw_x, raw_y)
        fingertip = (int(round(smooth_x)), int(round(smooth_y)))

        erase_gesture = is_open_hand(landmarks)
        pen_ratio = compute_pen_ratio(landmarks, width, height)
        if erase_gesture:
            # 지우기 제스처 중에는 펜을 내리지 않는다.
            self._pen_state.reset()
            pen_down = False
        else:
            pen_down = self._pen_state.update(pen_ratio)

        return PenFrame(
            hand_detected=True,
            fingertip=fingertip,
            raw_fingertip=(raw_x, raw_y),
            raw_z=raw_z,
            pen_down=pen_down,
            erase_gesture=erase_gesture,
            pen_ratio=pen_ratio,
        )

    def reset(self) -> None:
        """필터·펜 상태를 초기화한다."""
        self._smoother.reset()
        self._pen_state.reset()

    def close(self) -> None:
        self._tracker.close()

    def __enter__(self) -> "PenTracker":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
