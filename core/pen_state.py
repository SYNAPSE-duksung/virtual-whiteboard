"""검지 랜드마크 기반 pen-up/down 및 지우기 제스처 판정 (휴리스틱).

입력 랜드마크는 ``HandTracker``가 반환하는 정규화 좌표 배열
(shape ``(21, 3)``, 각 행 ``(x, y, z)``, 값 범위 0~1)을 전제로 한다.
CV 좌표만 다루며 UI/렌더링에 의존하지 않는다.

랜드마크 인덱스 상수는 ``core.distances``가 단일 소스이며, 기존 import 경로
(``from core.pen_state import INDEX_TIP`` 등)를 유지하기 위해 여기서 재수출한다.
"""

from __future__ import annotations

import math

import numpy as np

from core.distances import (
    FINGER_TIP_PIP_PAIRS,
    INDEX_DIP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_MCP,
    WRIST,
)

__all__ = [
    "WRIST",
    "INDEX_PIP",
    "INDEX_DIP",
    "INDEX_TIP",
    "MIDDLE_MCP",
    "FINGER_TIP_PIP_PAIRS",
    "compute_pen_ratio",
    "is_open_hand",
    "PenStateDetector",
]


def compute_pen_ratio(normalized_landmarks: np.ndarray, width: int, height: int) -> float:
    """검지 Tip(8)–DIP(7) y좌표 차이를 손 크기(wrist–middle_mcp)로 정규화한 비율.

    값이 작을수록 손가락이 눌린(pen-down) 상태로 판단한다. 손 크기가 0에 가까우면 1.0 반환.

    NOTE: 3주차 "정규화 수식 검증"의 **베이스라인**이다. 이 함수의 출력은
    ``tools/extract_landmarks.py``가 만드는 ``_coords.csv``의 ``pen_ratio`` 컬럼에
    그대로 들어가므로, 수식을 바꾸면 이미 추출한 CSV와 값이 어긋난다. 대안 수식은
    이 함수를 고치지 말고 ``core.distances``를 써서 별도 함수로 추가해 비교할 것.
    """
    lm = normalized_landmarks
    tip = lm[INDEX_TIP]
    dip = lm[INDEX_DIP]
    wrist = lm[WRIST]
    mid_mcp = lm[MIDDLE_MCP]

    hand_size = math.hypot((wrist[0] - mid_mcp[0]) * width, (wrist[1] - mid_mcp[1]) * height)
    if hand_size < 1e-6:
        return 1.0

    y_diff = abs((tip[1] - dip[1]) * height)
    return float(y_diff / hand_size)


def is_open_hand(normalized_landmarks: np.ndarray) -> bool:
    """네 손가락 끝이 모두 PIP보다 위(y가 작음)면 손을 편 것으로 간주 (지우기 제스처)."""
    lm = normalized_landmarks
    return all(lm[tip][1] < lm[pip][1] for tip, pip in FINGER_TIP_PIP_PAIRS)


class PenStateDetector:
    """pen_ratio 히스테리시스 상태 머신.

    down 진입은 낮은 임계값(``down_thresh``), up 복귀는 높은 임계값(``up_thresh``)을 써서
    임계값 부근에서의 채터링(빠른 on/off 반복)을 억제한다.
    """

    def __init__(self, *, down_thresh: float = 0.55, up_thresh: float = 0.70) -> None:
        if down_thresh > up_thresh:
            raise ValueError("down_thresh must be <= up_thresh for stable hysteresis")
        self.down_thresh = float(down_thresh)
        self.up_thresh = float(up_thresh)
        self._pen_down = False

    @property
    def pen_down(self) -> bool:
        return self._pen_down

    def update(self, pen_ratio: float) -> bool:
        """새 pen_ratio로 상태를 갱신하고 현재 pen-down 여부를 반환."""
        if self._pen_down:
            self._pen_down = pen_ratio < self.up_thresh
        else:
            self._pen_down = pen_ratio < self.down_thresh
        return self._pen_down

    def reset(self) -> None:
        """트래킹 손실·지우기 제스처 시 호출: pen-up으로 되돌린다."""
        self._pen_down = False
