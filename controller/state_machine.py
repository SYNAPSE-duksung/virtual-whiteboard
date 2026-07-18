"""``core.PenTracker``의 순간 판정 위에 얹는 시간 기반 안정화 계층.

``PenTracker.process()``가 매 프레임 내놓는 ``PenFrame``(순간 pen-down/erase 판정,
히스테리시스는 이미 pen_ratio 레벨에서 적용됨)을 그대로 렌더링에 쓰면 1~2프레임의
오판만으로도 획이 끊기거나 점이 튀고, 손 펴기 오검출 한 프레임에 캔버스가 지워질 수
있다. ``PenStateMachine``은 down/up 전환에 확정 시간을 요구하고(디바운싱), 짧은
트래킹 손실은 획을 유지한 채 흡수하며(홀드), 지우기는 일정 시간 이상 제스처가
유지돼야 1회성으로 발화시킨다.

``core/`` 모듈은 mediapipe에 의존하므로(이 프로젝트의 일부 개발 환경에는 미설치)
런타임에 import하지 않는다. ``PenFrame``은 타입 힌트로만 참조하고, 실제로는
``hand_detected``/``fingertip``/``pen_down``/``erase_gesture`` 속성을 덕 타이핑으로
읽는다 (``core.pen_tracker.PenFrame``과 동일한 속성을 가진 어떤 객체도 받아들인다).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.pen_tracker import PenFrame

__all__ = ["PenDecision", "PenStateMachine"]


@dataclass(frozen=True, slots=True)
class PenDecision:
    """``PenStateMachine.update()``가 한 프레임마다 내놓는 안정화된 판정."""

    pen_down: bool  # 안정화된 펜 상태
    draw_point: tuple[int, int] | None  # 이번 프레임에 획을 이을 점 (None = 그리지 않음)
    erase: bool  # 디바운스를 통과한 지우기 트리거 (1회성, 발화 프레임에만 True)
    erase_progress: float  # 지우기 확정까지 진행률 0.0~1.0 (게이지 표시용, 미진행 시 0.0)
    status: str  # "PEN DOWN" | "PEN UP" | "ERASE" | "HAND LOST"
    pending: str | None  # 전환 대기 설명 (디버그용)


class PenStateMachine:
    """``PenFrame`` 순간 판정을 디바운싱·홀드·지우기 확정 로직으로 안정화한다.

    내부 상태: 안정 pen-down 여부와 각 전환 스트릭(down 확정 대기/up 확정 대기/
    손실 허용/지우기 확인)의 시작 시각. 프레임 간 상태를 유지하므로 인스턴스를
    세션(또는 모드) 하나당 하나씩 재사용해야 하고, 모드 전환 시에는 ``reset()``을
    호출해 이전 모드의 진행 중이던 전환이 새 모드로 번지지 않게 한다.
    """

    def __init__(
        self,
        *,
        down_confirm_sec: float = 0.066,
        up_confirm_sec: float = 0.15,
        loss_tolerance_sec: float = 0.10,
        erase_confirm_sec: float = 0.25,
    ) -> None:
        self.down_confirm_sec = float(down_confirm_sec)
        self.up_confirm_sec = float(up_confirm_sec)
        self.loss_tolerance_sec = float(loss_tolerance_sec)
        self.erase_confirm_sec = float(erase_confirm_sec)

        self._stable_down = False
        self._loss_since: float | None = None
        self._down_streak_since: float | None = None
        self._up_streak_since: float | None = None
        self._erase_since: float | None = None
        self._erase_fired = False

    def reset(self) -> None:
        """모든 내부 상태를 초기화한다 (모드 전환 시 세션이 호출)."""
        self._stable_down = False
        self._loss_since = None
        self._down_streak_since = None
        self._up_streak_since = None
        self._erase_since = None
        self._erase_fired = False

    def update(self, frame: "PenFrame", *, timestamp: float | None = None) -> PenDecision:
        """``PenFrame`` 하나를 안정화해 ``PenDecision``으로 반환한다.

        ``timestamp``는 단조 증가 초 단위 시계 기준(기본 ``time.perf_counter()``).
        테스트/오프라인 재생에서는 33ms 간격 등 가짜 타임스탬프를 주입해 결정적으로
        재현할 수 있다.
        """
        t = time.perf_counter() if timestamp is None else timestamp

        # 1. 트래킹 손실.
        if not frame.hand_detected:
            return self._handle_loss(t)

        # 2. 손 재검출: 손실 시각 클리어 (홀드 중이었다면 안정 down이 그대로 유지된다).
        self._loss_since = None

        # 3. 지우기 게이트.
        if frame.erase_gesture:
            return self._handle_erase(frame, t)
        if self._erase_since is not None:
            # 제스처 해제: 스트릭·발화 플래그 리셋 (다시 펴면 재발화 가능).
            self._erase_since = None
            self._erase_fired = False

        # 4. 펜 디바운싱.
        return self._instant_pen_update(bool(frame.pen_down), frame.fingertip, t)

    def _handle_loss(self, t: float) -> PenDecision:
        # 손실 중에는 down/up/erase 스트릭을 모두 리셋한다.
        self._down_streak_since = None
        self._up_streak_since = None
        self._erase_since = None
        self._erase_fired = False

        if not self._stable_down:
            self._loss_since = None
            return PenDecision(False, None, False, 0.0, "HAND LOST", None)

        if self._loss_since is None:
            self._loss_since = t
        elapsed = t - self._loss_since
        if elapsed <= self.loss_tolerance_sec:
            # 홀드: 획을 유지한 채(그리지는 않고) 안정 down을 유지한다.
            return PenDecision(True, None, False, 0.0, "PEN DOWN", "손실 허용 중")

        self._stable_down = False
        return PenDecision(False, None, False, 0.0, "HAND LOST", None)

    def _handle_erase(self, frame: "PenFrame", t: float) -> PenDecision:
        if self._erase_since is None:
            self._erase_since = t
            self._erase_fired = False
        elapsed = t - self._erase_since

        if self._erase_fired:
            # 이미 발화한 제스처가 계속되는 중: 재발화하지 않는다.
            return PenDecision(False, None, False, 1.0, "ERASE", None)

        if elapsed >= self.erase_confirm_sec:
            # 확정 프레임: 1회성으로 발화하고 안정 상태를 up으로 강제한다(획 종료).
            self._erase_fired = True
            self._stable_down = False
            self._down_streak_since = None
            self._up_streak_since = None
            return PenDecision(False, None, True, 1.0, "ERASE", None)

        # 확정 전: 펜 로직은 순간 up으로 취급해 진행 중이던 down/up 전환을 계속 계산한다.
        progress = elapsed / self.erase_confirm_sec if self.erase_confirm_sec > 0 else 1.0
        partial = self._instant_pen_update(False, frame.fingertip, t)
        return PenDecision(
            partial.pen_down, partial.draw_point, False, progress, "ERASE", "지우기 확인 중"
        )

    def _instant_pen_update(
        self, instant_down: bool, fingertip: tuple[int, int] | None, t: float
    ) -> PenDecision:
        if not self._stable_down:
            # 안정 up.
            if not instant_down:
                self._down_streak_since = None
                return PenDecision(False, None, False, 0.0, "PEN UP", None)

            if self._down_streak_since is None:
                self._down_streak_since = t
            elapsed = t - self._down_streak_since
            if elapsed < self.down_confirm_sec:
                return PenDecision(False, None, False, 0.0, "PEN UP", "down 확정 대기")

            # 확정: 안정 down 진입, 이 프레임에서 획을 시작한다.
            self._stable_down = True
            self._down_streak_since = None
            self._up_streak_since = None
            draw_point = fingertip if fingertip is not None else None
            return PenDecision(True, draw_point, False, 0.0, "PEN DOWN", None)

        # 안정 down.
        if instant_down:
            self._up_streak_since = None
            draw_point = fingertip if fingertip is not None else None
            return PenDecision(True, draw_point, False, 0.0, "PEN DOWN", None)

        if self._up_streak_since is None:
            self._up_streak_since = t
        elapsed = t - self._up_streak_since
        if elapsed <= self.up_confirm_sec:
            # 홀드: 획을 유지한 채(그리지는 않고) up 확정을 기다린다.
            return PenDecision(True, None, False, 0.0, "PEN DOWN", "up 확정 대기")

        self._stable_down = False
        self._up_streak_since = None
        return PenDecision(False, None, False, 0.0, "PEN UP", None)
