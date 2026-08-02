"""4점 캘리브레이션 및 원근 변환 (4주차 A파트).

책상 위 필기 영역의 네 모서리(사선 각도로 촬영된 이미지의 픽셀 좌표)를 지정하면, 그
사각형을 위에서 내려다본 것처럼 반듯하게 펴는 투시 변환
(``cv2.getPerspectiveTransform`` + ``cv2.warpPerspective``)을 계산·저장·재사용한다.

좌표계
------
- **원본(src)**: 카메라 프레임 픽셀 좌표. 사선 각도로 찍혀 필기 영역이 사다리꼴로 보인다.
- **정류(rectified/dst)**: 보정 후 "가상 페이지" 픽셀 좌표. ``(0,0)``=좌상단,
  ``(dst_width,dst_height)``=우하단인 반듯한 직사각형. 이 좌표계에서는 원근 왜곡이 없어
  손끝 위치·펜 궤적을 실제 종이 위 위치처럼 다룰 수 있다.

점은 항상 **TL, TR, BR, BL** (좌상단부터 시계방향) 순서다 — OpenCV 표준 순서와 같아
``getPerspectiveTransform``에 그대로 전달할 수 있다.

4주차 시점의 위치: ``core.PenTracker``가 ``PerspectiveCalibration.contains()``로
손끝이 캘리브레이션한 사각형(필기 영역) 안인지를 매 프레임 게이팅에 쓴다(평면 밖이면
pen_ratio 판정과 무관하게 강제 pen up) — 자세한 한계는 ``core/pen_tracker.py`` docstring
참고. **이 게이팅은 2D 평면 범위 판정일 뿐, 진짜 3D 높이(접촉) 판정이 아니다** — 단일
RGB 카메라 + 4점 클릭만으로는 카메라 내부 파라미터·실측 평면 크기 없이 높이를 복원할 수
없다. 사선 각도에서 손이 회전하면 ``compute_pen_ratio`` 자체가 흔들린다는 문제
(``tools/validate_normalization.py``)는 아직 정류 좌표로 해소하지 않았다 — 5주차 강화
후보로 남아 있다.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

# 이미지가 이 4점보다 좁으면 클릭 실수(같은 지점을 두 번 찍음 등)로 간주한다.
MIN_QUAD_AREA_PX = 400.0
# 두 점이 이보다 가까우면 사실상 같은 점으로 취급해 즉시 거부한다.
MIN_POINT_SEPARATION_PX = 5.0

DEFAULT_CALIBRATION_PATH = Path("output/calibration.json")

Point = tuple[float, float]


class CalibrationError(ValueError):
    """4점 입력이 기하학적으로 유효하지 않을 때(퇴화·자기교차 등) 발생."""


def _as_array(points: Sequence[Point]) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.shape != (4, 2):
        raise CalibrationError(f"expected 4 points (x, y) each: shape={arr.shape}")
    return arr


def _quad_area(points: np.ndarray) -> float:
    """Shoelace 공식으로 사각형 면적(부호 없음)을 계산."""
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _validate_points(points: np.ndarray) -> None:
    """퇴화(중복점)·자기교차(잘못된 클릭 순서)·면적 과소를 걸러낸다."""
    for i in range(4):
        for j in range(i + 1, 4):
            if math.dist(points[i], points[j]) < MIN_POINT_SEPARATION_PX:
                # 이 메시지는 controller/main.py의 cv2 오버레이에 그대로 표시된다 —
                # Hershey 폰트에 한글 글리프가 없어 ASCII로만 작성한다.
                raise CalibrationError(
                    f"points {i} and {j} are too close "
                    f"({math.dist(points[i], points[j]):.1f}px) - reselect."
                )

    area = _quad_area(points)
    if area < MIN_QUAD_AREA_PX:
        raise CalibrationError(
            f"quad area too small ({area:.1f}px^2) - check click positions."
        )

    # TL,TR,BR,BL 순서라면 사각형은 볼록(convex)이어야 한다. 순서를 잘못 클릭해
    # 자기교차(나비꼴)가 생기면 isContourConvex가 False를 반환한다.
    contour = points.astype(np.int32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour):
        raise CalibrationError(
            "points do not form a convex quad - reclick in TL->TR->BR->BL "
            "(clockwise) order (looks self-intersecting)."
        )


def estimate_dst_size(points: Sequence[Point]) -> tuple[int, int]:
    """src 사각형의 변 길이로부터 자연스러운 정류 캔버스 크기(px)를 추정한다.

    상/하변 중 더 긴 쪽을 폭으로, 좌/우변 중 더 긴 쪽을 높이로 삼아 원본 필기 영역의
    비율을 최대한 보존한다(임의로 정사각형 등에 욱여넣지 않음).
    """
    arr = _as_array(points)
    tl, tr, br, bl = arr
    width = max(math.dist(tl, tr), math.dist(bl, br))
    height = max(math.dist(tl, bl), math.dist(tr, br))
    return max(int(round(width)), 10), max(int(round(height)), 10)


@dataclass(frozen=True)
class CalibrationPoints:
    """TL, TR, BR, BL 순서의 4점. 원본(src) 좌표계 전제."""

    top_left: Point
    top_right: Point
    bottom_right: Point
    bottom_left: Point

    def __post_init__(self) -> None:
        _validate_points(self.as_array())

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [self.top_left, self.top_right, self.bottom_right, self.bottom_left],
            dtype=np.float64,
        )

    @classmethod
    def from_sequence(cls, points: Sequence[Point]) -> "CalibrationPoints":
        """리스트/튜플([TL, TR, BR, BL]) → ``CalibrationPoints``. UI가 주로 이 형태로 넘긴다."""
        arr = _as_array(points)
        return cls(
            top_left=tuple(arr[0]),
            top_right=tuple(arr[1]),
            bottom_right=tuple(arr[2]),
            bottom_left=tuple(arr[3]),
        )


@dataclass(frozen=True)
class PerspectiveCalibration:
    """4점 캘리브레이션으로부터 계산한 투시 변환 행렬과 좌표 매핑.

    ``matrix``/``matrix_inv``는 ``__post_init__``에서 ``src_points``·``dst_size``로부터
    파생되므로 직접 만들 때는 ``PerspectiveCalibration.from_points()``를 쓸 것.
    """

    src_points: CalibrationPoints
    dst_size: tuple[int, int]
    matrix: np.ndarray = field(compare=False)
    matrix_inv: np.ndarray = field(compare=False)
    created_at: float = field(default_factory=time.time)
    source_frame_size: tuple[int, int] | None = None  # (width, height); 참고용, 필수 아님
    #: 캘리브레이션한 사각형의 **실제 물리 크기** (가로, 세로) 밀리미터.
    #: 3D 접촉 판정(``core.contact3d``)에서 평면 자세를 풀 때 필수다 — 이 값이 틀리면
    #: 복원 높이가 통째로 그 비율만큼 틀어진다(영점 보정으로 대부분 흡수되지만).
    #: A4 용지를 책상에 놓고 그 모서리를 찍으면 (297, 210)을 그대로 쓸 수 있다.
    plane_size_mm: tuple[float, float] | None = None

    @classmethod
    def from_points(
        cls,
        points: Sequence[Point] | CalibrationPoints,
        *,
        dst_size: tuple[int, int] | None = None,
        source_frame_size: tuple[int, int] | None = None,
        plane_size_mm: tuple[float, float] | None = None,
    ) -> "PerspectiveCalibration":
        """4점(+선택적 목적 크기)으로부터 투시 변환을 계산한다.

        ``dst_size``를 지정하지 않으면 원본 사각형의 변 길이로부터 자동 추정한다
        (``estimate_dst_size``). ``plane_size_mm``은 3D 접촉 판정용 실제 물리 크기다.
        """
        cal_points = (
            points if isinstance(points, CalibrationPoints) else CalibrationPoints.from_sequence(points)
        )
        src = cal_points.as_array()
        resolved_dst = dst_size or estimate_dst_size(src)
        width, height = resolved_dst
        if width <= 0 or height <= 0:
            raise CalibrationError(f"dst_size는 양수여야 합니다: {resolved_dst}")

        dst = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        src_f32 = src.astype(np.float32)
        matrix = cv2.getPerspectiveTransform(src_f32, dst)
        matrix_inv = cv2.getPerspectiveTransform(dst, src_f32)
        return cls(
            src_points=cal_points,
            dst_size=(width, height),
            matrix=matrix,
            matrix_inv=matrix_inv,
            source_frame_size=source_frame_size,
            plane_size_mm=tuple(float(v) for v in plane_size_mm) if plane_size_mm else None,
        )

    # ------------------------------------------------------------------
    # 프레임/좌표 변환
    # ------------------------------------------------------------------
    def warp_frame(self, frame: np.ndarray) -> np.ndarray:
        """카메라 프레임을 정류(bird's-eye) 뷰로 변환한 이미지를 반환."""
        return cv2.warpPerspective(frame, self.matrix, self.dst_size)

    def to_rectified(self, point: Point) -> Point:
        """원본(src) 픽셀 좌표 1개 → 정류(dst) 좌표."""
        return tuple(self.to_rectified_many([point])[0])

    def to_rectified_many(self, points: Sequence[Point]) -> np.ndarray:
        """원본 좌표 배열 ``(N, 2)`` → 정류 좌표 배열 ``(N, 2)``."""
        return self._apply(points, self.matrix)

    def from_rectified(self, point: Point) -> Point:
        """정류(dst) 좌표 1개 → 원본(src) 픽셀 좌표 (역변환)."""
        return tuple(self.from_rectified_many([point])[0])

    def from_rectified_many(self, points: Sequence[Point]) -> np.ndarray:
        """정류 좌표 배열 → 원본 좌표 배열 (역변환)."""
        return self._apply(points, self.matrix_inv)

    @staticmethod
    def _apply(points: Sequence[Point], matrix: np.ndarray) -> np.ndarray:
        arr = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(arr, matrix)
        return transformed.reshape(-1, 2)

    def contains(self, point: Point) -> bool:
        """원본(src) 좌표가 캘리브레이션한 필기 영역(사각형) 내부인지 판정.

        경계 위(0)도 내부로 취급한다. 손끝이 지정한 책상 영역을 벗어났을 때 판정을
        무시하는 등, 향후 pen-down 판정 게이팅에 쓸 수 있다.
        """
        contour = self.src_points.as_array().astype(np.float32).reshape(-1, 1, 2)
        return cv2.pointPolygonTest(contour, tuple(map(float, point)), False) >= 0

    # ------------------------------------------------------------------
    # 직렬화
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "version": 2,
            "src_points": self.src_points.as_array().tolist(),  # [[x,y]×4], TL,TR,BR,BL
            "dst_size": list(self.dst_size),
            "created_at": self.created_at,
            "source_frame_size": list(self.source_frame_size) if self.source_frame_size else None,
            "plane_size_mm": list(self.plane_size_mm) if self.plane_size_mm else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PerspectiveCalibration":
        # version 1 파일에는 plane_size_mm이 없다 — None으로 읽혀 3D 판정만 비활성화되고
        # 기존 2D 게이팅은 그대로 동작한다(하위 호환).
        cal = cls.from_points(
            data["src_points"],
            dst_size=tuple(data["dst_size"]),
            source_frame_size=tuple(data["source_frame_size"]) if data.get("source_frame_size") else None,
            plane_size_mm=tuple(data["plane_size_mm"]) if data.get("plane_size_mm") else None,
        )
        if "created_at" in data:
            object.__setattr__(cal, "created_at", data["created_at"])
        return cal

    def save(self, path: str | Path = DEFAULT_CALIBRATION_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CALIBRATION_PATH) -> "PerspectiveCalibration":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


class CalibrationPicker:
    """4점 선택 상태 머신 (TL→TR→BR→BL).

    마우스 클릭 좌표를 받아 누적하는 순수 상태 객체로, cv2/디스플레이에 의존하지 않는다
    (헤드리스 환경에서도 단위 테스트 가능). ``tools/calibrate.py``(독립 캘리브레이션 도구)와
    ``controller/main.py``(세션에 내장된 캘리브레이션 모드)가 이 클래스를 공유해, 두 진입점의
    4점 선택 로직이 갈라지지 않게 한다.
    """

    # cv2 오버레이(controller/main.py, tools/calibrate.py)에 그대로 표시되므로 ASCII로만
    # 작성한다 — Hershey 폰트에 한글 글리프가 없어 한글을 넣으면 ?????로 깨진다.
    LABELS: tuple[str, ...] = ("TL(top-left)", "TR(top-right)", "BR(bottom-right)", "BL(bottom-left)")

    def __init__(self) -> None:
        self._points: list[Point] = []

    @property
    def points(self) -> list[Point]:
        return list(self._points)

    @property
    def count(self) -> int:
        return len(self._points)

    @property
    def is_complete(self) -> bool:
        return len(self._points) == 4

    @property
    def next_label(self) -> str | None:
        """다음에 찍어야 할 점의 이름. 이미 완료됐으면 None."""
        return None if self.is_complete else self.LABELS[len(self._points)]

    def add_point(self, x: float, y: float) -> bool:
        """점 추가. 이미 4점이면 무시(추가 클릭 무시). 반환값은 완료 여부."""
        if self.is_complete:
            return True
        self._points.append((float(x), float(y)))
        return self.is_complete

    def undo(self) -> None:
        if self._points:
            self._points.pop()

    def reset(self) -> None:
        self._points.clear()

    def try_build(
        self,
        *,
        dst_size: tuple[int, int] | None = None,
        source_frame_size: tuple[int, int] | None = None,
        plane_size_mm: tuple[float, float] | None = None,
    ) -> tuple["PerspectiveCalibration | None", str | None]:
        """4점이 완료됐으면 변환을 계산해 ``(calibration, None)``, 아니면 ``(None, 이유)``.

        기하학적으로 유효하지 않은 4점(중복·자기교차 등)은 예외 대신 오류 메시지로 반환해
        UI에서 그대로 표시할 수 있게 한다.
        """
        if not self.is_complete:
            return None, None
        try:
            cal = PerspectiveCalibration.from_points(
                self._points,
                dst_size=dst_size,
                source_frame_size=source_frame_size,
                plane_size_mm=plane_size_mm,
            )
            return cal, None
        except CalibrationError as exc:
            return None, str(exc)
