"""Main control loop: webcam -> hand tracking -> stroke rendering.

``WhiteboardSession``이 트래킹·펜 판정(AUTO/MANUAL)·캔버스·CSV 기록·4점 캘리브레이션
평면 게이팅을 관리한다. 아키텍처·모드 설명·키 조작표·실행 옵션은 ``controller/README.md``
참고.

실행: ``python -m controller.main`` (옵션: ``--camera``, ``--mirror``, ``--calibration``)
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from controller.state_machine import PenStateMachine
from core.camera import DEFAULT_INTRINSICS_PATH, CameraIntrinsics
from core.contact3d import DEFAULT_PLANE_SIZE_MM
from core.geometry import DEFAULT_CALIBRATION_PATH, CalibrationPicker, PerspectiveCalibration
from core.pen_tracker import PenTracker
from core.recorder import CoordRecorder, CoordSample
from core.touch_calibration import (
    DEFAULT_SAMPLE_SECONDS,
    TouchCalibrationError,
    TouchSampleCollector,
    estimate_thresholds,
)
from ui.canvas import StrokeCanvas

_STATUS_COLORS = {
    "PEN DOWN": (0, 200, 0),
    "PEN UP": (200, 200, 200),
    "ERASE": (0, 215, 255),
    "HAND LOST": (0, 100, 255),
}

_CAL_POINT_COLOR = (0, 215, 255)
_CAL_LINE_COLOR = (60, 220, 60)
_CAL_TEXT_COLOR = (255, 255, 255)
_CAL_OK_COLOR = (0, 200, 0)
_CAL_ERR_COLOR = (0, 0, 255)
_PLANE_OUTLINE_COLOR = (255, 120, 0)
_CAL_CURSOR_COLOR = (255, 180, 0)
_TOUCH_BAR_BG = (70, 70, 70)
_TOUCH_BAR_FG = (0, 210, 210)
# 3D 디버그 오버레이 색 (BGR)
_DEBUG_NORMAL_COLOR = (255, 200, 0)  # 수선 벡터 (비접촉)
_DEBUG_CONTACT_COLOR = (0, 220, 60)  # 수선 벡터 (접촉) / down 문턱
_DEBUG_UP_COLOR = (0, 165, 255)  # up 문턱
_DEBUG_RULER_COLOR = (200, 200, 200)  # 높이 눈금
# 높이 게이지 범위(mm). 접촉 문턱(기본 12/22mm)과 손을 살짝 든 정도(~60mm)가 모두 들어오고,
# 영점이 덜 맞았을 때의 음수 높이도 보이도록 아래로 조금 여유를 둔다.
_GAUGE_MIN_MM, _GAUGE_MAX_MM = -20.0, 80.0
_GAUGE_TICKS_MM = (-20.0, 0.0, 20.0, 40.0, 60.0, 80.0)


def _px(point) -> tuple[int, int]:
    """부동소수 이미지 좌표를 OpenCV 그리기용 정수 픽셀로."""
    return (int(round(float(point[0]))), int(round(float(point[1]))))


@dataclass(frozen=True, slots=True)
class SessionDebug:
    """UI 디버그 패널용 세션 상태 스냅샷 (``WhiteboardSession.debug``)."""

    mode: str  # "AUTO" | "MANUAL"
    hand_detected: bool
    pen_ratio: float | None  # 마지막 프레임의 순간 pen_ratio
    down_thresh: float
    up_thresh: float
    instant_pen_down: bool  # PenTracker 순간 판정
    stable_pen_down: bool  # 상태머신 안정 판정 (수동 모드에선 _pen_requested)
    status: str
    pending: str | None  # 수동 모드에선 None
    erase_progress: float  # 수동 모드에선 0.0
    has_calibration: bool
    is_calibrating: bool
    in_bounds: bool | None  # None=캘리브레이션 미설정. 캘리브레이션 모드 중엔 None
    height_mm: float | None = None  # 3D 복원 높이(영점 보정 후). 3D 비활성 시 None
    contact_source: str = "ratio"  # "3d" | "ratio" — 이번 프레임 판정에 쓰인 신호
    has_3d_contact: bool = False  # 3D 접촉 판정 사용 가능 여부
    debug_3d: bool = False  # 3D 디버그 오버레이 표시 여부
    raw_height_mm: float | None = None  # 영점 보정 전 원시 높이
    contact_down_mm: float | None = None  # 3D 접촉 문턱(내림), 3D 비활성 시 None
    contact_up_mm: float | None = None  # 3D 접촉 문턱(올림)
    zero_offset_mm: float | None = None  # 학습된 영점 (표면을 짚었을 때의 원시 높이)


class WhiteboardSession:
    """
    한 번의 실행 동안 트래커, 캔버스 및 펜 상태를 관리합니다.

    `process_frame``은 프레임별 단일 진입점이므로 OpenCV 데모와 PyQt 앱(ui/app.py)은 
    완전히 동일한 파이프라인을 공유.
    """

    def __init__(
        self,
        *,
        line_color: tuple[int, int, int] = (0, 0, 255),
        line_thickness: int = 4,
        auto_mode: bool = True,
        min_cutoff: float = 1.0,
        beta: float = 0.3,
        pen_down_thresh: float = 0.55,
        pen_up_thresh: float = 0.70,
        down_confirm_sec: float = 0.066,
        up_confirm_sec: float = 0.15,
        loss_tolerance_sec: float = 0.10,
        erase_confirm_sec: float = 0.25,
        output_dir: str | Path = "output",
        calibration_path: str | Path | None = DEFAULT_CALIBRATION_PATH,
        touch_sample_seconds: float = DEFAULT_SAMPLE_SECONDS,
        intrinsics_path: str | Path | None = DEFAULT_INTRINSICS_PATH,
        plane_size_mm: tuple[float, float] = DEFAULT_PLANE_SIZE_MM,
        frame_size: tuple[int, int] = (1280, 720),
        use_3d_contact: bool = True,
    ) -> None:
        # 세션 시작 시 저장된 캘리브레이션을 1회 자동 로드한다. 없거나(최초 실행)
        # 손상됐으면 조용히 게이팅 없는 상태로 시작하고, 데모 루프가 `needs_calibration`을
        # 보고 최초 1회 지정을 강제한다. calibration_path=None이면 이 기능 자체를 끈다.
        self._calibration_path = calibration_path
        self._calibration: PerspectiveCalibration | None = None
        if calibration_path is not None:
            try:
                self._calibration = PerspectiveCalibration.load(calibration_path)
            except (FileNotFoundError, OSError, ValueError, KeyError, TypeError):
                self._calibration = None

        # 3D 접촉 판정용 카메라 내부 파라미터. 저장된 체스보드 캘리브레이션이 있으면 쓰고,
        # 없으면 화각 가정으로 근사한다(근사여도 영점 보정이 계통 오차를 대부분 흡수한다).
        # 실제 프레임 크기는 첫 process_frame에서 확정되므로 그때 다시 스케일한다.
        self._intrinsics_path = intrinsics_path
        self._plane_size_mm = tuple(float(v) for v in plane_size_mm)
        self._use_3d_contact = use_3d_contact
        self._intrinsics = (
            CameraIntrinsics.load_or_estimate(frame_size[0], frame_size[1], path=intrinsics_path)
            if use_3d_contact
            else None
        )

        self._tracker = PenTracker(
            min_cutoff=min_cutoff,
            beta=beta,
            pen_down_thresh=pen_down_thresh,
            pen_up_thresh=pen_up_thresh,
            calibration=self._calibration,
            intrinsics=self._intrinsics,
            use_3d_contact=use_3d_contact,
        )
        self._pen_down_thresh = pen_down_thresh
        self._pen_up_thresh = pen_up_thresh
        self._state_machine = PenStateMachine(
            down_confirm_sec=down_confirm_sec,
            up_confirm_sec=up_confirm_sec,
            loss_tolerance_sec=loss_tolerance_sec,
            erase_confirm_sec=erase_confirm_sec,
        )
        self._recorder = CoordRecorder(output_dir)
        self._line_color = line_color
        self._line_thickness = line_thickness
        self._canvas: StrokeCanvas | None = None
        # 펜 판정 모드: True=PenTracker 자동 판정(+손 펴기 지우기), False=키보드 수동.
        self._auto_mode = auto_mode
        # 수동 모드에서 SPACE/버튼으로 켜는 펜 상태.
        self._pen_requested = False
        self._status = "PEN UP"
        # 디버그 패널용: 마지막으로 처리한 PenFrame/PenDecision (처리 전에는 None).
        self._last_frame = None
        self._last_decision = None
        self._last_frame_size: tuple[int, int] | None = None

        # 캘리브레이션 지정(4점) 진행 상태. 마우스 클릭뿐 아니라 손끝으로도 점을 찍을 수
        # 있도록, 캘리브레이션 중에도 트래커를 계속 돌려 최근 손끝 위치를 들고 있는다.
        self._calibrating = False
        self._calibration_picker: CalibrationPicker | None = None
        self._calibration_message: str | None = None
        self._calibration_message_ok = True
        self._calibration_fingertip: tuple[int, int] | None = None

        # 터치(pen_ratio down/up) 임계값 캘리브레이션 진행 상태 — hover 단계 -> touch 단계
        # 순으로 표본을 모아 이 사람·이 카메라 구도에 맞는 임계값을 재추정한다.
        self._touch_sample_seconds = touch_sample_seconds
        self._touch_calibrating = False
        self._touch_phase: str | None = None  # "hover" | "touch" | None
        self._touch_phase_started_at: float | None = None
        self._touch_hover = TouchSampleCollector()
        self._touch_touch = TouchSampleCollector()
        # 같은 hover/touch 유지 동작에서 3D 복원 높이도 함께 모은다 — 한 번의 캘리브레이션이
        # pen_ratio 임계값과 3D 영점·접촉 문턱을 동시에 잡는다.
        self._touch_hover_mm = TouchSampleCollector()
        self._touch_touch_mm = TouchSampleCollector()
        self._touch_message: str | None = None
        self._touch_message_ok = True

        # 3D 디버그 오버레이(수선 벡터 + 높이 눈금 + 수치 패널) 표시 여부.
        self._debug_3d = False

    @property
    def pen_requested(self) -> bool:
        return self._pen_requested

    @property
    def auto_mode(self) -> bool:
        return self._auto_mode

    @property
    def mode_name(self) -> str:
        return "AUTO" if self._auto_mode else "MANUAL"

    @property
    def status(self) -> str:
        return self._status

    @property
    def canvas(self) -> StrokeCanvas | None:
        return self._canvas

    @property
    def debug(self) -> SessionDebug:
        """UI 디버그 패널용 스냅샷. 아직 프레임을 처리하지 않았으면 초기값을 반환한다."""
        frame = self._last_frame
        decision = self._last_decision
        hand_detected = frame.hand_detected if frame is not None else False
        pen_ratio = frame.pen_ratio if frame is not None else None
        instant_pen_down = frame.pen_down if frame is not None else False
        in_bounds = frame.in_bounds if frame is not None else None
        if self._auto_mode:
            stable_pen_down = decision.pen_down if decision is not None else False
            pending = decision.pending if decision is not None else None
            erase_progress = decision.erase_progress if decision is not None else 0.0
        else:
            stable_pen_down = self._pen_requested
            pending = None
            erase_progress = 0.0
        estimator = self._tracker.contact_estimator
        detector = self._tracker.contact_detector
        return SessionDebug(
            mode=self.mode_name,
            hand_detected=hand_detected,
            pen_ratio=pen_ratio,
            down_thresh=self._pen_down_thresh,
            up_thresh=self._pen_up_thresh,
            instant_pen_down=instant_pen_down,
            stable_pen_down=stable_pen_down,
            status=self._status,
            pending=pending,
            erase_progress=erase_progress,
            has_calibration=self._calibration is not None,
            is_calibrating=self._calibrating,
            in_bounds=in_bounds,
            height_mm=frame.height_mm if frame is not None else None,
            contact_source=frame.contact_source if frame is not None else "ratio",
            has_3d_contact=self._tracker.has_3d_contact,
            debug_3d=self._debug_3d,
            raw_height_mm=frame.raw_height_mm if frame is not None else None,
            contact_down_mm=detector.down_mm if estimator is not None else None,
            contact_up_mm=detector.up_mm if estimator is not None else None,
            zero_offset_mm=estimator.zero_offset_mm if estimator is not None else None,
        )

    # ------------------------------------------------------------------
    # 3D 접촉 판정 (카메라 intrinsics + 평면 실제 크기)
    # ------------------------------------------------------------------
    @property
    def has_3d_contact(self) -> bool:
        """3D 높이 기반 접촉 판정을 쓸 수 있는 상태인지."""
        return self._tracker.has_3d_contact

    @property
    def debug_3d(self) -> bool:
        """3D 디버그 오버레이가 켜져 있는지."""
        return self._debug_3d

    def set_debug_3d(self, enabled: bool) -> None:
        """3D 디버그 오버레이를 켜고 끈다 (꺼져 있으면 계산 자체를 하지 않는다)."""
        self._debug_3d = bool(enabled)
        self._tracker.set_debug_3d(self._debug_3d)

    def toggle_debug_3d(self) -> bool:
        """디버그 오버레이를 토글하고 결과 상태를 반환한다."""
        self.set_debug_3d(not self._debug_3d)
        return self._debug_3d

    @property
    def intrinsics(self) -> CameraIntrinsics | None:
        return self._intrinsics

    @property
    def plane_size_mm(self) -> tuple[float, float]:
        return self._plane_size_mm

    def _rescale_intrinsics(self, width: int, height: int) -> None:
        """실제 카메라 해상도가 확정되면 intrinsics를 그 해상도로 다시 맞춘다."""
        if self._intrinsics is None:
            return
        rescaled = self._intrinsics.scaled_to(width, height)
        if rescaled is self._intrinsics:
            return
        self._intrinsics = rescaled
        self._tracker.set_intrinsics(rescaled)

    # ------------------------------------------------------------------
    # 캘리브레이션 (4점 지정 → 평면 게이팅)
    # ------------------------------------------------------------------
    @property
    def has_calibration(self) -> bool:
        return self._calibration is not None

    @property
    def needs_calibration(self) -> bool:
        """세션 시작 시 유효한 캘리브레이션이 없어 최초 1회 지정이 필요한 상태인지."""
        return self._calibration is None

    @property
    def is_calibrating(self) -> bool:
        return self._calibrating

    @property
    def calibration(self) -> PerspectiveCalibration | None:
        return self._calibration

    @property
    def calibration_picker(self) -> CalibrationPicker | None:
        return self._calibration_picker

    @property
    def calibration_message(self) -> tuple[str, bool] | None:
        """``(메시지, 성공여부)``. 표시할 메시지가 없으면 ``None``."""
        if self._calibration_message is None:
            return None
        return (self._calibration_message, self._calibration_message_ok)

    @property
    def calibration_fingertip(self) -> tuple[int, int] | None:
        """캘리브레이션 모드 중 마지막으로 추적된 손끝 픽셀 좌표 (없으면 None)."""
        return self._calibration_fingertip

    def start_calibration(self) -> None:
        """캘리브레이션 모드로 진입한다. 이미 캘리브레이션이 있어도 재지정할 수 있다
        (``confirm_calibration()``으로 확정하기 전까지 기존 캘리브레이션은 그대로 유효).

        터치 캘리브레이션과 동시에 진행할 수 없다 — 진행 중이면 무시된다.
        """
        if self._touch_calibrating:
            return
        self._calibrating = True
        self._calibration_picker = CalibrationPicker()
        self._calibration_message = None
        self._calibration_message_ok = True
        self._calibration_fingertip = None
        self._tracker.reset()  # 캘리브레이션 이전 필터/히스테리시스 상태가 커서에 섞이지 않게.
        if self._canvas is not None:
            self._canvas.pen_up()

    def add_calibration_point(self, x: float, y: float) -> None:
        """캘리브레이션 모드일 때만 좌표를 점으로 추가한다 (그 외엔 무시). 마우스 클릭용."""
        if not self._calibrating or self._calibration_picker is None:
            return
        self._calibration_picker.add_point(x, y)
        self._calibration_message = None

    def add_calibration_point_at_fingertip(self) -> bool:
        """추적 중인 손끝의 현재 위치에 캘리브레이션 점을 찍는다.

        마우스 없이 **손가락으로 직접 4점을 지정**할 수 있게 하는 방법 — 손끝을 원하는
        모서리 위에 놓고 이 메서드를 호출한다(데모 루프에서는 SPACE 키). 캘리브레이션
        모드가 아니거나 손이 검출되지 않았으면 아무 것도 하지 않고 ``False``를 반환한다.
        """
        if not self._calibrating or self._calibration_fingertip is None:
            return False
        self.add_calibration_point(*self._calibration_fingertip)
        return True

    def undo_calibration_point(self) -> None:
        if self._calibrating and self._calibration_picker is not None:
            self._calibration_picker.undo()
            self._calibration_message = None

    def reset_calibration_points(self) -> None:
        if self._calibrating and self._calibration_picker is not None:
            self._calibration_picker.reset()
            self._calibration_message = None

    def confirm_calibration(self) -> bool:
        """4점이 유효하면 저장·적용하고 캘리브레이션 모드를 종료한다.

        반환값은 성공 여부. 실패(미완료·퇴화 4점) 시 ``calibration_message``에 이유가
        남고 캘리브레이션 모드는 유지된다(재시도 가능).
        """
        if not self._calibrating or self._calibration_picker is None:
            return False
        cal, err = self._calibration_picker.try_build(
            source_frame_size=self._last_frame_size,
            # 3D 접촉 판정에 필요한 평면 실제 크기를 함께 기록한다 (없으면 3D 비활성).
            plane_size_mm=self._plane_size_mm if self._use_3d_contact else None,
        )
        if cal is None:
            self._calibration_message = err or "4점을 먼저 모두 지정하세요."
            self._calibration_message_ok = False
            return False

        if self._calibration_path is not None:
            cal.save(self._calibration_path)
        self._calibration = cal
        self._tracker.set_calibration(cal)  # 내부적으로 필터·펜 상태도 함께 리셋된다.
        self._state_machine.reset()
        self._calibrating = False
        self._calibration_picker = None
        self._calibration_fingertip = None
        mode_3d = "3D 접촉 ON" if self._tracker.has_3d_contact else "3D 접촉 OFF(pen_ratio 폴백)"
        self._calibration_message = f"캘리브레이션 적용됨 (dst={cal.dst_size}, {mode_3d})"
        self._calibration_message_ok = True
        if self._canvas is not None:
            self._canvas.pen_up()
        self._status = "PEN UP"
        return True

    def cancel_calibration(self) -> None:
        """캘리브레이션 모드를 취소한다. 기존 캘리브레이션(있었다면)은 그대로 유지된다."""
        self._calibrating = False
        self._calibration_picker = None
        self._calibration_message = None
        self._calibration_fingertip = None
        self._tracker.reset()
        self._state_machine.reset()

    # ------------------------------------------------------------------
    # 터치(pen_ratio down/up) 임계값 캘리브레이션
    # ------------------------------------------------------------------
    @property
    def touch_sample_seconds(self) -> float:
        return self._touch_sample_seconds

    @property
    def is_touch_calibrating(self) -> bool:
        return self._touch_calibrating

    @property
    def touch_phase(self) -> str | None:
        """``"hover"`` | ``"touch"`` | ``None``(진행 중 아님)."""
        return self._touch_phase

    @property
    def touch_message(self) -> tuple[str, bool] | None:
        """``(메시지, 성공여부)``. 표시할 메시지가 없으면 ``None``."""
        if self._touch_message is None:
            return None
        return (self._touch_message, self._touch_message_ok)

    @property
    def touch_progress(self) -> float:
        """현재 단계의 진행률 0.0~1.0 (진행 중 아니면 0.0)."""
        if not self._touch_calibrating or self._touch_phase_started_at is None:
            return 0.0
        elapsed = time.perf_counter() - self._touch_phase_started_at
        return min(elapsed / self._touch_sample_seconds, 1.0) if self._touch_sample_seconds > 0 else 1.0

    def start_touch_calibration(self) -> None:
        """터치 임계값 캘리브레이션을 시작한다: hover 단계부터.

        4점 캘리브레이션과 동시에 진행할 수 없다 — 진행 중이면 무시된다.
        """
        if self._calibrating:
            return
        self._touch_calibrating = True
        self._touch_phase = "hover"
        self._touch_phase_started_at = time.perf_counter()
        self._touch_hover.reset()
        self._touch_touch.reset()
        self._touch_hover_mm.reset()
        self._touch_touch_mm.reset()
        self._touch_message = None
        self._touch_message_ok = True
        self._tracker.reset()
        if self._canvas is not None:
            self._canvas.pen_up()

    def cancel_touch_calibration(self) -> None:
        """터치 캘리브레이션을 취소한다. 기존 임계값은 그대로 유지된다."""
        self._touch_calibrating = False
        self._touch_phase = None
        self._touch_phase_started_at = None
        self._touch_message = None
        self._tracker.reset()
        self._state_machine.reset()

    def _advance_touch_calibration(
        self, pen_ratio: float | None, raw_height_mm: float | None = None
    ) -> None:
        """터치 캘리브레이션 진행 중 프레임 하나를 처리 — 표본 수집 + 단계 전환."""
        now = time.perf_counter()
        started = self._touch_phase_started_at or now
        elapsed = now - started

        if self._touch_phase == "hover":
            self._touch_hover.add(pen_ratio)
            self._touch_hover_mm.add(raw_height_mm)
            if elapsed >= self._touch_sample_seconds:
                self._touch_phase = "touch"
                self._touch_phase_started_at = now
        elif self._touch_phase == "touch":
            self._touch_touch.add(pen_ratio)
            self._touch_touch_mm.add(raw_height_mm)
            if elapsed >= self._touch_sample_seconds:
                self._finish_touch_calibration()

    def _calibrate_3d_contact(self) -> str | None:
        """모은 높이 표본으로 3D 영점과 접촉 문턱을 잡는다. 결과 요약 문구(또는 None).

        touch 단계 높이의 중앙값이 곧 "표면을 짚었을 때의 측정값" = 영점이다. 이걸 빼면
        계통 오차(intrinsics·손 모델 스케일 오차)가 대부분 상쇄된다.
        """
        if not self._tracker.has_3d_contact:
            return None
        hover_mm = self._touch_hover_mm.samples
        touch_mm = self._touch_touch_mm.samples
        try:
            result = estimate_thresholds(hover_mm, touch_mm)
        except TouchCalibrationError:
            return None

        zero_offset = result.touch_median  # 표면 접촉 시 측정값을 0으로.
        self._tracker.set_contact_zero_offset(zero_offset)
        self._tracker.set_contact_thresholds(
            down_mm=result.down_thresh - zero_offset,
            up_mm=result.up_thresh - zero_offset,
        )
        noise = float(np.std(touch_mm)) if len(touch_mm) > 1 else float("nan")
        separation = result.hover_median - result.touch_median
        return (
            f"3D: 영점={zero_offset:.1f}mm, 문턱={result.down_thresh - zero_offset:.1f}/"
            f"{result.up_thresh - zero_offset:.1f}mm, 분리={separation:.1f}mm, 접촉노이즈σ={noise:.1f}mm"
        )

    def _finish_touch_calibration(self) -> None:
        """두 신호(pen_ratio·3D 높이)를 **독립적으로** 보정하고, 하나라도 성공하면 성공 처리.

        pen_ratio가 hover/touch를 구분하지 못하는 상황이야말로 3D로 넘어온 이유이므로,
        pen_ratio 실패가 3D 영점 보정까지 취소시키면 안 된다.
        """
        parts: list[str] = []
        failures: list[str] = []

        # 1) 3D 영점·접촉 문턱 (가능할 때).
        contact_summary = self._calibrate_3d_contact()
        if contact_summary:
            parts.append(contact_summary)
        elif self._tracker.has_3d_contact:
            failures.append("3D 높이 표본으로는 접촉/비접촉이 구분되지 않음")

        # 2) pen_ratio 임계값 (3D가 실패했을 때의 폴백 신호이므로 항상 함께 갱신 시도).
        try:
            result = estimate_thresholds(self._touch_hover.samples, self._touch_touch.samples)
        except TouchCalibrationError as exc:
            failures.append(f"pen_ratio: {exc}")
        else:
            self._pen_down_thresh = result.down_thresh
            self._pen_up_thresh = result.up_thresh
            self._tracker.set_thresholds(
                down_thresh=result.down_thresh, up_thresh=result.up_thresh
            )
            parts.append(
                f"pen_ratio 임계값: down={result.down_thresh:.3f} up={result.up_thresh:.3f}"
            )

        self._touch_calibrating = False
        self._touch_phase = None
        self._touch_phase_started_at = None
        self._state_machine.reset()

        if parts:
            self._touch_message = " | ".join(parts)
            self._touch_message_ok = True
        else:
            self._touch_message = " / ".join(failures) or "캘리브레이션에 실패했습니다."
            self._touch_message_ok = False
            self._tracker.reset()
            return

        if self._canvas is not None:
            self._canvas.pen_up()
        self._status = "PEN UP"

    def set_pen_down(self, down: bool) -> None:
        """수동 모드 펜 상태 설정 (자동 모드에서는 무시)."""
        self._pen_requested = down

    def toggle_pen(self) -> bool:
        self._pen_requested = not self._pen_requested
        return self._pen_requested

    def set_auto_mode(self, auto: bool) -> None:
        """펜 판정 모드 지정. 전환 시 진행 중이던 획·상태를 정리한다."""
        if auto == self._auto_mode:
            return
        self._auto_mode = auto
        self._reset_pen_transients()

    def toggle_mode(self) -> bool:
        """자동/수동 모드를 전환하고 새 모드가 자동인지 반환."""
        self._auto_mode = not self._auto_mode
        self._reset_pen_transients()
        return self._auto_mode

    def _reset_pen_transients(self) -> None:
        """모드 전환 시 이전 모드의 펜 상태가 새 모드로 번지지 않도록 초기화."""
        self._pen_requested = False
        self._tracker.reset()
        self._state_machine.reset()
        if self._canvas is not None:
            self._canvas.pen_up()
        self._status = "PEN UP"

    def clear(self) -> None:
        if self._canvas is not None:
            self._canvas.clear()

    def save_canvas(self, path: str | Path) -> Path | None:
        """Write the white-background canvas to ``path``; None if empty."""
        if self._canvas is None:
            return None
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), self._canvas.image)
        return path

    @property
    def is_recording(self) -> bool:
        return self._recorder.recording

    @property
    def recording_path(self) -> Path | None:
        return self._recorder.path

    def start_recording(self) -> Path:
        """좌표 CSV 기록을 시작하고 파일 경로를 반환한다."""
        return self._recorder.start()

    def stop_recording(self) -> None:
        self._recorder.stop()

    def toggle_recording(self) -> bool:
        """CSV 기록을 켜고/끄고, 켜졌으면 True를 반환한다."""
        return self._recorder.toggle()

    def process_frame(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Run tracking + stroke recording and return the annotated frame."""
        height, width = bgr_frame.shape[:2]
        if self._last_frame_size != (width, height):
            # 실제 카메라 해상도가 확정(또는 변경)되면 intrinsics를 그 해상도로 다시 맞춘다 —
            # 초점거리·주점이 픽셀 단위라 해상도가 다르면 복원 거리가 그 비율만큼 틀어진다.
            self._last_frame_size = (width, height)
            self._rescale_intrinsics(width, height)
        if self._canvas is None:
            self._canvas = StrokeCanvas(
                width,
                height,
                line_color=self._line_color,
                line_thickness=self._line_thickness,
            )

        if self._calibrating:
            # 캘리브레이션 중에는 캔버스에 그리거나 기록하지 않지만, 손끝으로도 점을 찍을
            # 수 있도록 트래킹 자체는 계속 돌려 커서 위치로 쓴다 (마우스 없이도 지정 가능).
            cal_result = self._tracker.process(bgr_frame)
            self._calibration_fingertip = cal_result.fingertip
            return self._draw_calibration_overlay(bgr_frame)

        if self._touch_calibrating:
            # 터치 캘리브레이션 중에도 캔버스에는 그리지 않고 pen_ratio 표본만 모은다.
            touch_result = self._tracker.process(bgr_frame)
            self._advance_touch_calibration(touch_result.pen_ratio, touch_result.raw_height_mm)
            return self._draw_touch_calibration_overlay(bgr_frame, touch_result.fingertip)

        result = self._tracker.process(bgr_frame)
        fingertip = result.fingertip
        self._last_frame = result

        if self._auto_mode:
            # 자동 모드: 손실 프레임도 포함해 항상 상태머신을 거친다 (손실 홀드는
            # PenStateMachine의 책임 — 여기서 손실을 먼저 걸러내면 브리징이 깨진다).
            decision = self._state_machine.update(result)
            self._last_decision = decision
            if decision.erase:
                self._canvas.clear()
            elif decision.draw_point is not None:
                self._draw_to(decision.draw_point)
            elif not decision.pen_down:
                self._canvas.pen_up()
            # else: 홀드(안정 down인데 이번 프레임은 그리지 않음) — 아무것도 하지 않는다.
            # 여기서 canvas.pen_up()을 부르면 진행 중 획이 끊겨 브리징(획 이어붙이기)이 깨진다.
            self._status = decision.status
        else:
            self._last_decision = None
            if not result.hand_detected:
                # Tracking loss must break the stroke, otherwise the next
                # detection draws a straight line from the stale point.
                self._canvas.pen_up()
                self._status = "HAND LOST"
            else:
                # 수동 모드: 키보드/버튼 펜만 사용 (지우기는 clear() = C 키).
                if self._pen_requested:
                    self._draw_to(fingertip)
                    self._status = "PEN DOWN"
                else:
                    self._canvas.pen_up()
                    self._status = "PEN UP"

        # 기록 중이면 매 프레임 CSV로 남긴다 (모드 무관, 기록 중이 아니면 no-op).
        self._recorder.write(CoordSample.from_pen_frame(result))

        annotated = self._canvas.overlay(bgr_frame)
        if self._calibration is not None:
            # 캘리브레이션한 필기 영역 경계를 항상 얇게 겹쳐 그려, 게이팅 기준을
            # 눈으로 확인할 수 있게 한다.
            outline = self._calibration.src_points.as_array().astype(np.int32)
            cv2.polylines(annotated, [outline], True, _PLANE_OUTLINE_COLOR, 1, cv2.LINE_AA)
        if fingertip is not None:
            cv2.circle(annotated, fingertip, 10, (255, 180, 0), 2)
        cv2.putText(
            annotated,
            self._status,
            (15, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            _STATUS_COLORS[self._status],
            2,
            cv2.LINE_AA,
        )
        mode_label = f"MODE: {self.mode_name}"
        if self._recorder.recording:
            mode_label += "  * REC"
        mode_label += "  CAL:OK" if self._calibration is not None else "  CAL:NONE"
        mode_label += "  3D" if self._tracker.has_3d_contact else "  2D"
        cv2.putText(
            annotated,
            mode_label,
            (15, height - 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0) if self._auto_mode else (180, 220, 255),
            2,
            cv2.LINE_AA,
        )
        if self._auto_mode and result.in_bounds is False:
            # 손은 검출됐지만 캘리브레이션한 평면 밖 — pen_ratio와 무관하게 up으로
            # 게이팅됐다는 것을 상태 텍스트만으로는 알기 어려워 별도로 표시한다.
            cv2.putText(
                annotated,
                "OUT OF PLANE",
                (15, height - 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                _CAL_ERR_COLOR,
                2,
                cv2.LINE_AA,
            )
        elif self._auto_mode and result.height_mm is not None:
            # 3D 판정 중이면 실제 복원 높이를 mm로 보여준다 — 판정 근거를 눈으로 확인.
            cv2.putText(
                annotated,
                f"H {result.height_mm:+6.1f}mm",
                (15, height - 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                _CAL_OK_COLOR if result.pen_down else _CAL_TEXT_COLOR,
                2,
                cv2.LINE_AA,
            )
        if self._debug_3d:
            self._draw_3d_debug(annotated, result)
        return annotated

    def _draw_3d_debug(self, annotated: np.ndarray, result) -> None:
        """손끝→평면 수선(직교 벡터)과 높이 눈금·수치를 겹쳐 그린다 (제자리 수정).

        3D가 꺼져 있거나 이번 프레임에서 자세 추정이 실패했으면 그 사실 자체를 표시한다 —
        디버그를 켰는데 아무것도 안 보이면 "왜 안 보이는지"를 알 수 없기 때문이다.
        """
        height, width = annotated.shape[:2]
        debug = getattr(result, "contact_debug", None)
        if debug is None:
            # OpenCV의 Hershey 폰트에는 한글 글리프가 없어 ????로 깨진다 — 오버레이 문구는
            # 전부 ASCII로 쓴다(코드베이스의 다른 오버레이도 같은 규칙).
            reason = (
                "3D DEBUG: OFF - need 4-pt calibration + plane size + intrinsics (press K)"
                if not self._tracker.has_3d_contact
                else "3D DEBUG: hand pose failed this frame (falling back to pen_ratio)"
            )
            cv2.putText(annotated, reason, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, _CAL_ERR_COLOR, 2, cv2.LINE_AA)
            return

        tip = np.array(debug.tip_px, dtype=np.float64)
        foot = np.array(debug.foot_px, dtype=np.float64)
        contact = result.pen_down
        axis_color = _DEBUG_CONTACT_COLOR if contact else _DEBUG_NORMAL_COLOR

        # 수선(평면에 직교하는 벡터): 수선의 발 → 손끝. 이게 곧 "평면에서 얼마나 떴는가"다.
        cv2.line(annotated, _px(foot), _px(tip), axis_color, 2, cv2.LINE_AA)
        # 수선의 발 = 손끝을 평면에 정사영한 위치. 평면 위에 있음을 십자로 표시.
        fx, fy = _px(foot)
        cv2.line(annotated, (fx - 9, fy), (fx + 9, fy), axis_color, 1, cv2.LINE_AA)
        cv2.line(annotated, (fx, fy - 9), (fx, fy + 9), axis_color, 1, cv2.LINE_AA)
        cv2.circle(annotated, _px(tip), 5, axis_color, -1, cv2.LINE_AA)

        # 눈금을 그릴 방향: 수선의 화면상 방향에 수직한 단위벡터.
        axis = tip - foot
        norm = float(np.linalg.norm(axis))
        perp = np.array([-axis[1], axis[0]]) / norm if norm > 1e-6 else np.array([1.0, 0.0])

        for mm, px in debug.ruler_px:
            p = np.array(px, dtype=np.float64)
            cv2.line(annotated, _px(p - perp * 6), _px(p + perp * 6),
                     _DEBUG_RULER_COLOR, 1, cv2.LINE_AA)
            cv2.putText(annotated, f"{mm:.0f}", _px(p + perp * 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, _DEBUG_RULER_COLOR, 1, cv2.LINE_AA)

        # 접촉 문턱은 눈금보다 굵고 길게 — 손끝이 이 선 아래로 내려가면 pen down이다.
        for label, mm, px in debug.threshold_px:
            p = np.array(px, dtype=np.float64)
            color = _DEBUG_CONTACT_COLOR if label == "down" else _DEBUG_UP_COLOR
            cv2.line(annotated, _px(p - perp * 14), _px(p + perp * 14), color, 2, cv2.LINE_AA)
            cv2.putText(annotated, f"{label} {mm:.0f}", _px(p - perp * 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        estimator = self._tracker.contact_estimator
        detector = self._tracker.contact_detector
        plane_x, plane_y = debug.plane_xy_mm
        plane_w, plane_h = self._plane_size_mm
        zero = "n/a  (press T)" if estimator is None else f"{estimator.zero_offset_mm:+8.1f} mm"
        lines = [
            ("3D DEBUG   (D to hide)", _CAL_TEXT_COLOR),
            (f"height     {debug.height_mm:+8.1f} mm", axis_color),
            (f"raw height {debug.raw_height_mm:+8.1f} mm", _CAL_TEXT_COLOR),
            (f"zero offs  {zero}",
             _CAL_TEXT_COLOR if estimator is not None else _CAL_ERR_COLOR),
            (f"thresh d/u {detector.down_mm:.1f} / {detector.up_mm:.1f} mm", _DEBUG_RULER_COLOR),
            (f"plane XY   {plane_x:6.0f},{plane_y:6.0f} of {plane_w:.0f}x{plane_h:.0f}"
             + ("" if debug.inside_plane else "  OUT"),
             _CAL_TEXT_COLOR if debug.inside_plane else _CAL_ERR_COLOR),
            (f"hand reproj{debug.reprojection_error_px:7.1f} px", _CAL_TEXT_COLOR),
            (f"source {result.contact_source}   contact {'YES' if contact else 'no'}", axis_color),
        ]
        # 좌하단 상태 문구와 겹치지 않도록 우상단에 패널을 띄운다.
        panel_w, line_h = 300, 23
        x0 = max(width - panel_w - 12, 15)
        y0 = 18
        overlay = annotated.copy()
        cv2.rectangle(overlay, (x0 - 10, y0 - 6),
                      (x0 + panel_w + 2, y0 + line_h * len(lines)), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0, annotated)
        for i, (text, color) in enumerate(lines):
            cv2.putText(annotated, text, (x0, y0 + line_h * (i + 1) - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.47, color, 1, cv2.LINE_AA)

        self._draw_height_gauge(annotated, debug.height_mm, detector, contact)

    def _draw_height_gauge(self, annotated: np.ndarray, height_mm: float, detector, contact: bool) -> None:
        """화면 왼쪽에 높이 막대 게이지를 그린다 (투영과 무관한 읽기 전용 눈금).

        카메라가 책상을 거의 수직으로 내려다보면 평면의 법선이 시선과 거의 나란해져,
        화면에 투영된 수선이 몇 픽셀로 찌부러져 눈으로 읽을 수 없다. 이 게이지는 3D
        기하를 투영하지 않고 높이 값만 막대로 그리므로 그 구도에서도 항상 읽힌다.
        """
        frame_h = annotated.shape[0]
        x, top, bottom = 40, 120, frame_h - 140
        span = bottom - top
        lo, hi = _GAUGE_MIN_MM, _GAUGE_MAX_MM

        def y_of(mm: float) -> int:
            # 위쪽이 높은 높이(hi), 아래쪽이 lo. 범위를 벗어나면 끝에 붙인다.
            frac = (float(mm) - lo) / (hi - lo)
            return int(round(bottom - max(0.0, min(1.0, frac)) * span))

        overlay = annotated.copy()
        cv2.rectangle(overlay, (x - 16, top - 22), (x + 78, bottom + 20), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)
        cv2.line(annotated, (x, top), (x, bottom), _DEBUG_RULER_COLOR, 1, cv2.LINE_AA)
        cv2.putText(annotated, "mm", (x - 12, top - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, _DEBUG_RULER_COLOR, 1, cv2.LINE_AA)

        for mm in _GAUGE_TICKS_MM:
            y = y_of(mm)
            cv2.line(annotated, (x - 5, y), (x + 5, y), _DEBUG_RULER_COLOR, 1, cv2.LINE_AA)
            cv2.putText(annotated, f"{mm:+.0f}", (x + 10, y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, _DEBUG_RULER_COLOR, 1, cv2.LINE_AA)

        # 접촉 문턱: 이 두 선 사이가 히스테리시스 구간이다.
        for mm, color in ((detector.down_mm, _DEBUG_CONTACT_COLOR),
                          (detector.up_mm, _DEBUG_UP_COLOR)):
            y = y_of(mm)
            cv2.line(annotated, (x - 12, y), (x + 12, y), color, 2, cv2.LINE_AA)

        # 현재 높이 표시자.
        y = y_of(height_mm)
        color = _DEBUG_CONTACT_COLOR if contact else _DEBUG_NORMAL_COLOR
        cv2.line(annotated, (x - 14, y), (x + 14, y), color, 2, cv2.LINE_AA)
        cv2.circle(annotated, (x, y), 5, color, -1, cv2.LINE_AA)
        cv2.putText(annotated, f"{height_mm:+.1f}", (x - 14, bottom + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def _draw_to(self, fingertip: tuple[int, int]) -> None:
        """펜을 내린 상태로 손끝 위치까지 획을 잇는다."""
        if self._canvas is None:
            return
        if self._canvas.is_pen_down:
            self._canvas.move(fingertip)
        else:
            self._canvas.pen_down(fingertip)

    def _draw_calibration_overlay(self, frame: np.ndarray) -> np.ndarray:
        """캘리브레이션 모드의 점·연결선·손끝 커서·안내 문구·메시지를 그려 반환한다."""
        annotated = frame.copy()
        picker = self._calibration_picker
        if picker is None:
            return annotated

        points = picker.points
        for i, (x, y) in enumerate(points):
            pt = (int(round(x)), int(round(y)))
            cv2.circle(annotated, pt, 8, _CAL_POINT_COLOR, -1)
            cv2.putText(
                annotated, str(i + 1), (pt[0] + 10, pt[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, _CAL_POINT_COLOR, 2, cv2.LINE_AA,
            )
        if len(points) >= 2:
            pts_arr = np.array([[int(x), int(y)] for x, y in points], dtype=np.int32)
            cv2.polylines(annotated, [pts_arr], picker.is_complete, _CAL_LINE_COLOR, 2, cv2.LINE_AA)

        # 손끝 커서: 마우스 없이도 손가락으로 점을 찍을 수 있다는 것을 보여준다.
        if self._calibration_fingertip is not None:
            cv2.circle(annotated, self._calibration_fingertip, 12, _CAL_CURSOR_COLOR, 2)
            cv2.circle(annotated, self._calibration_fingertip, 2, _CAL_CURSOR_COLOR, -1)

        if not picker.is_complete:
            guide = f"[캘리브레이션] 다음: {picker.next_label} ({picker.count}/4) — 클릭 또는 손끝 위치에서 SPACE"
        else:
            guide = "[캘리브레이션] 4점 완료 — Enter/S 적용 / Z 취소 / X 리셋 / Esc 취소"
        # main()의 FPS 카운터가 (15,30)을 이미 쓰므로 그 아래에 그린다.
        cv2.putText(annotated, guide, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, _CAL_TEXT_COLOR, 2, cv2.LINE_AA)
        if self._calibration_fingertip is None:
            cv2.putText(
                annotated, "(손이 검출되지 않아 SPACE로는 찍을 수 없음 — 클릭은 가능)",
                (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 255), 1, cv2.LINE_AA,
            )

        if self._calibration_message:
            color = _CAL_OK_COLOR if self._calibration_message_ok else _CAL_ERR_COLOR
            cv2.putText(
                annotated, self._calibration_message, (15, annotated.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
            )
        return annotated

    def _draw_touch_calibration_overlay(
        self, frame: np.ndarray, fingertip: tuple[int, int] | None
    ) -> np.ndarray:
        """터치 캘리브레이션 모드의 단계 안내·진행률 게이지·손끝 커서를 그려 반환한다."""
        annotated = frame.copy()
        if fingertip is not None:
            cv2.circle(annotated, fingertip, 12, _CAL_CURSOR_COLOR, 2)

        if self._touch_phase == "hover":
            title = "[터치 캘리브레이션 1/2] 손가락을 화면 위에 띄운 채 가만히 두세요 (닿지 않게)"
        elif self._touch_phase == "touch":
            title = "[터치 캘리브레이션 2/2] 책상에 손가락을 대고 누른 상태로 가만히(또는 천천히) 유지하세요"
        else:
            title = "[터치 캘리브레이션]"
        cv2.putText(annotated, title, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _CAL_TEXT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(
            annotated, "Esc 취소", (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _CAL_TEXT_COLOR, 1, cv2.LINE_AA,
        )

        # 진행률 게이지 (0~1)
        bar_x, bar_y, bar_w, bar_h = 15, 100, 300, 14
        cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), _TOUCH_BAR_BG, -1)
        fill_w = int(bar_w * self.touch_progress)
        if fill_w > 0:
            cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), _TOUCH_BAR_FG, -1)
        cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), _CAL_TEXT_COLOR, 1)

        if self._touch_message:
            color = _CAL_OK_COLOR if self._touch_message_ok else _CAL_ERR_COLOR
            # 긴 결과 문구는 화면 폭에 맞춰 두 줄로 접는다.
            msg = self._touch_message
            mid = len(msg) // 2
            split = msg.rfind(" ", 0, mid + 20) if len(msg) > 55 else -1
            lines = [msg] if split == -1 else [msg[:split], msg[split + 1:]]
            base_y = annotated.shape[0] - 20 - 22 * (len(lines) - 1)
            for i, line in enumerate(lines):
                cv2.putText(
                    annotated, line, (15, base_y + 22 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
                )
        return annotated

    def close(self) -> None:
        self._recorder.close()
        self._tracker.close()

    def __enter__(self) -> "WhiteboardSession":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


_WINDOW_NAME = "Virtual Whiteboard - M mode / K calib / T touch-calib / SPACE pen / R rec / C clear / S save / Q quit"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="camera device index")
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="flip the frame horizontally (selfie view; off by default for the desk view)",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default=str(DEFAULT_CALIBRATION_PATH),
        help="캘리브레이션 저장/로드 경로 (기본 output/calibration.json)",
    )
    parser.add_argument(
        "--plane-size",
        type=float,
        nargs=2,
        metavar=("WIDTH_MM", "HEIGHT_MM"),
        default=list(DEFAULT_PLANE_SIZE_MM),
        help="캘리브레이션할 사각형의 실제 물리 크기(mm). 기본 297 210 = A4 가로. "
             "3D 접촉 판정의 정확도가 이 값에 비례하므로 실제로 찍는 영역과 맞출 것",
    )
    parser.add_argument(
        "--intrinsics",
        type=str,
        default=str(DEFAULT_INTRINSICS_PATH),
        help="카메라 내부 파라미터 JSON 경로. 없으면 화각 가정으로 근사한다",
    )
    parser.add_argument(
        "--no-3d",
        action="store_true",
        help="3D 접촉 판정을 끄고 기존 pen_ratio 휴리스틱만 사용",
    )
    parser.add_argument(
        "--debug-3d",
        action="store_true",
        help="3D 디버그 오버레이(평면 수선 벡터·높이 눈금·수치)를 켠 채로 시작 (실행 중 D로 토글)",
    )
    args = parser.parse_args()

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        print(f"카메라 {args.camera}을(를) 열 수 없습니다.")
        return 1

    print("M: 모드 전환 | K: 4점 캘리브레이션 | T: 접촉 캘리브레이션(3D 영점 포함) | D: 3D 디버그 표시 | R: CSV 기록 | C: 지우기 | S: 저장 | Q: 종료")
    if not args.no_3d:
        print(f"[3D] 평면 실제 크기 {args.plane_size[0]:.0f}x{args.plane_size[1]:.0f}mm 가정 "
              f"— 다르면 --plane-size 로 지정하세요 (A4 가로면 그대로 두면 됩니다)")
    previous_time = time.perf_counter()
    last_touch_message: tuple[str, bool] | None = None
    try:
        with WhiteboardSession(
            calibration_path=args.calibration,
            intrinsics_path=args.intrinsics,
            plane_size_mm=(args.plane_size[0], args.plane_size[1]),
            use_3d_contact=not args.no_3d,
        ) as session:
            if args.debug_3d:
                session.set_debug_3d(True)
            cv2.namedWindow(_WINDOW_NAME)
            cv2.setMouseCallback(
                _WINDOW_NAME,
                lambda event, x, y, flags, userdata: session.add_calibration_point(x, y)
                if event == cv2.EVENT_LBUTTONDOWN
                else None,
            )

            if session.needs_calibration:
                # 세션 시작 시 유효한 캘리브레이션이 없으면(최초 실행) 1회 강제한다.
                # 이후에는 K로 언제든 재캘리브레이션할 수 있다.
                print("[캘리브레이션] 저장된 캘리브레이션이 없어 최초 지정을 시작합니다.")
                print("  TL→TR→BR→BL 순서로 클릭 (Z 취소 / X 리셋 / Enter,S 적용 / Esc는 무시 — 최초 1회는 필수)")
                session.start_calibration()

            while True:
                ok, frame = camera.read()
                if not ok:
                    print("카메라 프레임을 읽지 못했습니다.")
                    return 1
                if args.mirror:
                    frame = cv2.flip(frame, 1)

                annotated = session.process_frame(frame)

                # 터치 캘리브레이션은 키 입력이 아니라 시간 경과로 자동 완료되므로,
                # 결과 메시지가 바뀔 때만 콘솔에도 한 번 출력한다 (화면 표시는 항상 됨).
                touch_msg = session.touch_message
                if touch_msg is not None and touch_msg != last_touch_message:
                    print(f"[터치 캘리브레이션] {touch_msg[0]}")
                last_touch_message = touch_msg

                current_time = time.perf_counter()
                fps = 1.0 / max(current_time - previous_time, 1e-6)
                previous_time = current_time
                cv2.putText(annotated, f"FPS {fps:.1f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow(_WINDOW_NAME, annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if session.is_calibrating:
                    if key == ord("z"):
                        session.undo_calibration_point()
                    elif key == ord("x"):
                        session.reset_calibration_points()
                    elif key == ord(" "):  # SPACE: 현재 손끝 위치에 점 찍기 (마우스 없이)
                        if not session.add_calibration_point_at_fingertip():
                            print("[캘리브레이션] 손이 검출되지 않아 점을 찍을 수 없습니다 — 클릭으로 지정하세요.")
                    elif key in (13, ord("s")):  # Enter 또는 S: 적용
                        if session.confirm_calibration():
                            cal = session.calibration
                            print(f"[캘리브레이션] 적용 완료 (dst={cal.dst_size if cal else '?'})")
                        else:
                            msg = session.calibration_message
                            print(f"[캘리브레이션] {msg[0] if msg else '4점을 모두 지정하세요.'}")
                    elif key == 27:  # Esc: 취소
                        if session.has_calibration:
                            session.cancel_calibration()
                            print("[캘리브레이션] 취소 — 기존 캘리브레이션 유지")
                        else:
                            print("[캘리브레이션] 아직 캘리브레이션이 없어 취소할 수 없습니다 — 4점을 지정하세요.")
                elif session.is_touch_calibrating:
                    if key == 27:  # Esc: 취소
                        session.cancel_touch_calibration()
                        print("[터치 캘리브레이션] 취소 — 기존 임계값 유지")
                elif key == ord("k"):
                    session.start_calibration()
                    print("[캘리브레이션] 시작 — 클릭 또는 손끝+SPACE로 TL→TR→BR→BL 순서 지정 (Z 취소 / X 리셋 / Enter,S 적용 / Esc 취소)")
                elif key == ord("t"):
                    session.start_touch_calibration()
                    print(f"[터치 캘리브레이션] 시작 — 먼저 {session.touch_sample_seconds:.1f}초간 손가락을 화면 위에 띄우세요 (Esc 취소)")
                elif key == ord("d"):
                    on = session.toggle_debug_3d()
                    print(f"[3D 디버그] {'표시' if on else '숨김'}"
                          + ("" if session.has_3d_contact else " (※ 3D 접촉 판정이 비활성 상태입니다)"))
                elif key == ord("m"):
                    auto = session.toggle_mode()
                    print(f"모드 전환: {'자동(손끝 판정)' if auto else '수동(SPACE 펜)'}")
                elif key == ord("r"):
                    if session.toggle_recording():
                        print(f"[CSV] 기록 시작: {session.recording_path}")
                    else:
                        print("[CSV] 기록 중지")
                elif key == ord(" "):
                    session.toggle_pen()
                elif key == ord("c"):
                    session.clear()
                elif key == ord("s"):
                    saved = session.save_canvas(
                        Path("captures") / time.strftime("canvas_%Y%m%d_%H%M%S.png")
                    )
                    if saved is not None:
                        print(f"캔버스 저장: {saved}")
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
