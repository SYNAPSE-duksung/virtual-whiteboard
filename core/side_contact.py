"""측면(폰) 카메라 기준 pen up/down 판정 — 듀얼 카메라 계획(CLAUDE.md 참고)의 1단계 구성요소.

노트북 카메라(사선/탑다운)의 3D 접촉 판정(``core/contact3d.py``)은 이중 ``solvePnP``로
손끝의 평면 위 실제 높이(mm)를 복원하는 정교한 방식이지만, 그 정확도는 하필 pen-down에
가까운 자세(손이 카메라 쪽으로 접히는 자세)에서 가장 나쁘다는 게 실측으로 드러났다.

측면(책상 높이에서 옆모습을 보는) 카메라는 문제 자체가 다르다 — 손끝-책상 간격이 원근
왜곡 없이 이미지에 거의 그대로 투영되므로, 4점 평면 캘리브레이션 + 이중 solvePnP 같은
무거운 기하 없이도 **책상면을 나타내는 직선 하나**만 잡으면 충분하다: 손끝에서 그 직선까지의
수직 거리(px)가 곧 "얼마나 떠 있는가"다. ``pen_ratio``·3D ``height_mm``과 방향을 맞춰
**값이 작을수록 접촉**으로 둔다.

손끝은 물리적으로 책상 위쪽에만 있으므로(책상을 뚫고 내려갈 수 없다) 부호 없는 거리만으로
충분하다.

``ContactDetector``(``core/contact3d.py``)와 히스테리시스 판정 로직이 사실상 같지만, 이
모듈 안에 독립적으로 다시 구현한다 — 4인 분업 환경에서 모듈 간 상호 의존을 최소화해
병합 충돌을 줄이기 위한 팀 컨벤션(CLAUDE.md "아키텍처 규칙" 참고)을 따른 것.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.pen_state import INDEX_TIP

DEFAULT_BASELINE_PATH = Path("output/side_baseline.json")

# 두 점이 이보다 가까우면 직선 방향이 노이즈에 흔들려 즉시 거부한다.
MIN_POINT_SEPARATION_PX = 20.0

# 접촉 판정 기본 문턱(px). 카메라-책상 거리·해상도에 따라 달라지므로 실측 후 조정 필요 —
# 임시값이며, hover/touch 표본으로 재추정하려면 core.touch_calibration.estimate_thresholds()를
# 이 신호(baseline까지 거리)에 그대로 적용하면 된다(범용 함수라 pen_ratio·height_mm과 공용).
DEFAULT_DOWN_PX = 20.0
DEFAULT_UP_PX = 35.0

Point = tuple[float, float]


class SideBaselineError(ValueError):
    """책상면 기준선 2점이 기하학적으로 유효하지 않을 때(퇴화) 발생."""


@dataclass(frozen=True)
class SideBaseline:
    """측면 카메라 이미지 위, 책상면을 나타내는 직선.

    ``point``는 직선 위의 한 점(픽셀), ``direction``은 그 직선의 단위 방향 벡터.
    """

    point: tuple[float, float]
    direction: tuple[float, float]

    def distance_of(self, image_point: Point) -> float:
        """이미지 위의 한 점에서 이 직선까지 수직 거리(px, 항상 ≥0)."""
        px, py = float(image_point[0]), float(image_point[1])
        ox, oy = self.point
        dx, dy = self.direction
        vx, vy = px - ox, py - oy
        return abs(vx * dy - vy * dx)  # |v × d̂| : 단위 방향벡터에 대한 수직 거리

    def to_dict(self) -> dict:
        return {"point": list(self.point), "direction": list(self.direction)}

    @classmethod
    def from_dict(cls, data: dict) -> "SideBaseline":
        return cls(point=tuple(data["point"]), direction=tuple(data["direction"]))

    def save(self, path: str | Path = DEFAULT_BASELINE_PATH) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path = DEFAULT_BASELINE_PATH) -> "SideBaseline":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


def estimate_baseline(p1: Point, p2: Point) -> SideBaseline:
    """책상 가장자리 위의 두 점(예: 앞쪽 모서리 양 끝)으로 기준선을 만든다."""
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    dx, dy = x2 - x1, y2 - y1
    length = float(np.hypot(dx, dy))
    if length < MIN_POINT_SEPARATION_PX:
        raise SideBaselineError(
            f"two points too close ({length:.1f}px) - reselect two farther points "
            "along the desk edge."
        )
    return SideBaseline(point=(x1, y1), direction=(dx / length, dy / length))


class SideBaselinePicker:
    """마우스 클릭 2번으로 기준선을 지정하는 상태 머신 (``core.geometry.CalibrationPicker`` 축소판)."""

    LABELS = ("start", "end")

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
        return len(self._points) == 2

    @property
    def next_label(self) -> str | None:
        return None if self.is_complete else self.LABELS[len(self._points)]

    def add_point(self, x: float, y: float) -> bool:
        """점 추가. 이미 2점이면 무시. 반환값은 완료 여부."""
        if self.is_complete:
            return True
        self._points.append((float(x), float(y)))
        return self.is_complete

    def undo(self) -> None:
        if self._points:
            self._points.pop()

    def reset(self) -> None:
        self._points.clear()

    def try_build(self) -> tuple[SideBaseline | None, str | None]:
        """2점이 완료됐으면 ``(baseline, None)``, 아니면 ``(None, 이유)``."""
        if not self.is_complete:
            return None, None
        try:
            return estimate_baseline(self._points[0], self._points[1]), None
        except SideBaselineError as exc:
            return None, str(exc)


def measure(hand, baseline: SideBaseline, *, landmark_index: int = INDEX_TIP) -> float:
    """``TrackedHand``에서 기준선까지 손끝의 거리(px)를 계산한다.

    서브픽셀 좌표(``pixel_landmarks_f``)가 있으면 우선 쓰고, 없으면 정수 픽셀로 대체한다
    (``core.contact3d.Contact3DEstimator.measure``와 동일한 이유 — 정수 반올림 양자화 회피).
    """
    points = getattr(hand, "pixel_landmarks_f", None)
    if points is None:
        points = hand.pixel_landmarks
    x, y = points[landmark_index]
    return baseline.distance_of((float(x), float(y)))


class SideContactDetector:
    """기준선까지 거리(px)에 대한 히스테리시스 접촉 판정. 값이 작을수록 접촉이다."""

    def __init__(self, *, down_px: float = DEFAULT_DOWN_PX, up_px: float = DEFAULT_UP_PX) -> None:
        if down_px > up_px:
            raise ValueError("down_px must be <= up_px for stable hysteresis")
        self.down_px = float(down_px)
        self.up_px = float(up_px)
        self._contact = False

    @property
    def contact(self) -> bool:
        return self._contact

    def update(self, distance_px: float) -> bool:
        if self._contact:
            self._contact = distance_px < self.up_px
        else:
            self._contact = distance_px < self.down_px
        return self._contact

    def set_thresholds(self, *, down_px: float, up_px: float) -> None:
        if down_px > up_px:
            raise ValueError("down_px must be <= up_px for stable hysteresis")
        self.down_px = float(down_px)
        self.up_px = float(up_px)

    def reset(self) -> None:
        self._contact = False
