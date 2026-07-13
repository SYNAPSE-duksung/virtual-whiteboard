"""검지 랜드마크 기반 pen-up/down 및 지우기 제스처 판정 (휴리스틱).

입력 랜드마크는 ``HandTracker``가 반환하는 정규화 좌표 배열
(shape ``(21, 3)``, 각 행 ``(x, y, z)``, 값 범위 0~1)을 전제로 한다.
CV 좌표만 다루며 UI/렌더링에 의존하지 않는다.
"""

from __future__ import annotations

import math

import numpy as np

# MediaPipe hand landmark 인덱스
WRIST = 0
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9

# 지우기(손 펴기) 제스처 판정용 (끝 landmark, PIP landmark) 쌍
FINGER_TIP_PIP_PAIRS: tuple[tuple[int, int], ...] = ((8, 6), (12, 10), (16, 14), (20, 18))


def compute_pen_ratio(normalized_landmarks: np.ndarray, width: int, height: int) -> float:
    """검지 Tip(8)–DIP(7) y좌표 차이를 손 크기(wrist–middle_mcp)로 정규화한 비율.

    값이 작을수록 손가락이 눌린(pen-down) 상태로 판단한다. 손 크기가 0에 가까우면 1.0 반환.
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
