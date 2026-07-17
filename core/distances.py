"""손가락 부위별 유클리드 거리 연산 및 정규화 (3주차 A파트).

MediaPipe 정규화 랜드마크 배열 ``(21, 3)``을 입력받아 임의의 랜드마크 쌍 사이 거리를
2D/3D로 계산하고, 손 크기로 정규화한다. 판정(pen_state)·렌더링과 무관한 순수 기하 모듈로,
``core.pen_state``가 이 모듈의 랜드마크 상수를 재사용한다.

좌표계 주의
-----------
MediaPipe 정규화 좌표는 x가 폭, y가 높이로 **각각 따로** 0~1 정규화되어 있어, 그대로
``hypot(dx, dy)``를 계산하면 화면 종횡비(예: 16:9)만큼 왜곡된 거리가 나온다. 따라서 이
모듈은 항상 픽셀 스케일로 되돌린 뒤(``x*width``, ``y*height``) 거리를 계산한다.

z는 손목을 원점으로 한 상대 깊이이며 "x와 대략 같은 스케일"이라는 MediaPipe 규약에 따라
``width``로 스케일한다. z는 x/y보다 노이즈가 크므로 기본값은 ``use_z=False``(2D)이고,
3D 거리는 Z축 임계값 실험에서 명시적으로 켜서 쓴다.
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# MediaPipe hand landmark 인덱스 (21개)
# ---------------------------------------------------------------------------
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

NUM_LANDMARKS = 21

LANDMARK_NAMES: tuple[str, ...] = (
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)

# 손가락별 마디 체인 (뿌리 → 끝). 인접 쌍이 곧 "부위별" 마디 구간이다.
FINGER_CHAINS: dict[str, tuple[int, ...]] = {
    "thumb": (THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP),
    "index": (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}

# 지우기(손 펴기) 제스처 판정용 (끝, PIP) 쌍 — pen_state가 재사용.
FINGER_TIP_PIP_PAIRS: tuple[tuple[int, int], ...] = (
    (INDEX_TIP, INDEX_PIP),
    (MIDDLE_TIP, MIDDLE_PIP),
    (RING_TIP, RING_PIP),
    (PINKY_TIP, PINKY_PIP),
)

# 손 크기 정규화 분모 후보. 3주차 "거리 정규화 수식 검증"에서 비교 대상이 된다.
#   wrist_middle_mcp : 기존 compute_pen_ratio 베이스라인
#   wrist_index_mcp  : 검지 뿌리 기준
#   palm_width       : 검지 MCP ↔ 새끼 MCP (손바닥 폭)
#   middle_finger    : 중지 마디 길이 합 (손가락 길이)
#   bbox_diagonal    : 전체 랜드마크 바운딩박스 대각선
HAND_SCALE_METHODS: tuple[str, ...] = (
    "wrist_middle_mcp",
    "wrist_index_mcp",
    "palm_width",
    "middle_finger",
    "bbox_diagonal",
)

DEFAULT_HAND_SCALE = "wrist_middle_mcp"

_EPS = 1e-6


# ---------------------------------------------------------------------------
# 내부 유틸
# ---------------------------------------------------------------------------
def scale_points(
    normalized_landmarks: np.ndarray,
    width: int,
    height: int,
    *,
    use_z: bool = False,
) -> np.ndarray:
    """정규화 좌표를 픽셀 스케일 3D 점 배열 ``(N, 3)``으로 되돌린다.

    ``use_z=False``면 z열은 0으로 채워져 사실상 2D 거리가 된다.
    """
    points = np.asarray(normalized_landmarks, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError(
            f"landmarks must be a (N, 2+) array, got shape {points.shape}"
        )
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    scaled = np.zeros((points.shape[0], 3), dtype=np.float64)
    scaled[:, 0] = points[:, 0] * width
    scaled[:, 1] = points[:, 1] * height
    if use_z:
        if points.shape[1] < 3:
            raise ValueError("use_z=True requires a z column in landmarks")
        # MediaPipe 규약: z의 스케일은 x와 대략 동일 → width로 스케일.
        scaled[:, 2] = points[:, 2] * width
    return scaled


def _chain_length(points: np.ndarray, chain: tuple[int, ...]) -> float:
    return float(
        sum(
            math.dist(points[a], points[b])
            for a, b in zip(chain, chain[1:])
        )
    )


# ---------------------------------------------------------------------------
# 프레임 단위 기하 계산기
# ---------------------------------------------------------------------------
class HandGeometry:
    """한 프레임의 랜드마크를 픽셀 스케일로 한 번만 변환해두고 거리 질의를 처리한다.

    한 프레임에서 여러 거리를 뽑을 때(부위별 마디 길이, feature 세트 등) 매번
    재스케일하지 않으므로 모듈 함수들보다 효율적이다.
    """

    __slots__ = ("_points", "use_z")

    def __init__(
        self,
        normalized_landmarks: np.ndarray,
        width: int,
        height: int,
        *,
        use_z: bool = False,
    ) -> None:
        self._points = scale_points(normalized_landmarks, width, height, use_z=use_z)
        self.use_z = use_z

    @property
    def points(self) -> np.ndarray:
        """픽셀 스케일 점 배열 ``(N, 3)`` (읽기 전용으로 취급할 것)."""
        return self._points

    def distance(self, a: int, b: int) -> float:
        """랜드마크 ``a``–``b`` 사이 유클리드 거리(픽셀)."""
        return float(math.dist(self._points[a], self._points[b]))

    def distances(self, pairs) -> dict[tuple[int, int], float]:
        """여러 쌍의 거리를 한 번에. ``{(a, b): 거리}``."""
        return {(a, b): self.distance(a, b) for a, b in pairs}

    def scale(self, method: str = DEFAULT_HAND_SCALE) -> float:
        """손 크기(정규화 분모)를 ``method`` 방식으로 계산."""
        if method == "wrist_middle_mcp":
            return self.distance(WRIST, MIDDLE_MCP)
        if method == "wrist_index_mcp":
            return self.distance(WRIST, INDEX_MCP)
        if method == "palm_width":
            return self.distance(INDEX_MCP, PINKY_MCP)
        if method == "middle_finger":
            return _chain_length(self._points, FINGER_CHAINS["middle"])
        if method == "bbox_diagonal":
            spans = self._points.max(axis=0) - self._points.min(axis=0)
            return float(np.linalg.norm(spans))
        raise ValueError(
            f"unknown hand scale method {method!r}; expected one of {HAND_SCALE_METHODS}"
        )

    def ratio(
        self,
        a: int,
        b: int,
        *,
        method: str = DEFAULT_HAND_SCALE,
        default: float = math.nan,
    ) -> float:
        """``a``–``b`` 거리를 손 크기로 나눈 정규화 비율 (해상도·손 크기 불변).

        손 크기가 0에 가까우면(랜드마크 붕괴) ``default``를 반환한다.
        """
        denominator = self.scale(method)
        if denominator < _EPS:
            return default
        return self.distance(a, b) / denominator

    def finger_segments(self, finger: str) -> dict[str, float]:
        """손가락 하나의 **부위별 마디 길이**. ``{"index_mcp->index_pip": 42.1, ...}``."""
        try:
            chain = FINGER_CHAINS[finger]
        except KeyError:
            raise ValueError(
                f"unknown finger {finger!r}; expected one of {tuple(FINGER_CHAINS)}"
            ) from None
        return {
            f"{LANDMARK_NAMES[a]}->{LANDMARK_NAMES[b]}": self.distance(a, b)
            for a, b in zip(chain, chain[1:])
        }

    def all_finger_segments(self) -> dict[str, float]:
        """다섯 손가락 전체의 부위별 마디 길이를 한 dict로."""
        lengths: dict[str, float] = {}
        for finger in FINGER_CHAINS:
            lengths.update(self.finger_segments(finger))
        return lengths


# ---------------------------------------------------------------------------
# 모듈 함수 (단발성 호출용 얇은 래퍼)
# ---------------------------------------------------------------------------
def euclidean(
    normalized_landmarks: np.ndarray,
    a: int,
    b: int,
    width: int,
    height: int,
    *,
    use_z: bool = False,
) -> float:
    """랜드마크 ``a``–``b`` 사이 유클리드 거리(픽셀)."""
    return HandGeometry(normalized_landmarks, width, height, use_z=use_z).distance(a, b)


def pairwise_distances(
    normalized_landmarks: np.ndarray,
    pairs,
    width: int,
    height: int,
    *,
    use_z: bool = False,
) -> dict[tuple[int, int], float]:
    """여러 랜드마크 쌍의 거리를 한 번에 계산."""
    return HandGeometry(normalized_landmarks, width, height, use_z=use_z).distances(pairs)


def hand_scale(
    normalized_landmarks: np.ndarray,
    width: int,
    height: int,
    *,
    method: str = DEFAULT_HAND_SCALE,
    use_z: bool = False,
) -> float:
    """손 크기(정규화 분모)를 계산."""
    return HandGeometry(normalized_landmarks, width, height, use_z=use_z).scale(method)


def normalized_distance(
    normalized_landmarks: np.ndarray,
    a: int,
    b: int,
    width: int,
    height: int,
    *,
    method: str = DEFAULT_HAND_SCALE,
    use_z: bool = False,
    default: float = math.nan,
) -> float:
    """``a``–``b`` 거리를 손 크기로 정규화한 비율."""
    return HandGeometry(normalized_landmarks, width, height, use_z=use_z).ratio(
        a, b, method=method, default=default
    )


def finger_segment_lengths(
    normalized_landmarks: np.ndarray,
    finger: str,
    width: int,
    height: int,
    *,
    use_z: bool = False,
) -> dict[str, float]:
    """손가락 하나의 부위별 마디 길이."""
    return HandGeometry(normalized_landmarks, width, height, use_z=use_z).finger_segments(
        finger
    )
