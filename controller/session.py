"""``WhiteboardSession`` — 트래킹·펜 판정(AUTO/MANUAL)·캔버스·4점 캘리브레이션 평면
게이팅·OCR/LLM 트리거를 관리하는 프레임워크-독립 세션 엔진.

``process_frame()``이 프레임별 단일 진입점이므로 OpenCV 데모(``controller/main.py``)와
PyQt 앱(``ui/app.py``)이 완전히 동일한 파이프라인을 공유한다. 프레임 오버레이 그리기는
``controller/overlay.py``에 분리돼 있다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from controller import overlay
from controller.ocr_llm_pipeline import OcrLlmPipelineWorker, PipelineResult
from controller.ocr_llm_pipeline import STATUS_IDLE as _PIPELINE_IDLE
from controller.state_machine import PenStateMachine
from core.camera import DEFAULT_INTRINSICS_PATH, CameraIntrinsics
from core.contact3d import DEFAULT_PLANE_SIZE_MM, MAX_REPROJECTION_ERROR_PX
from core.geometry import DEFAULT_CALIBRATION_PATH, CalibrationPicker, PerspectiveCalibration
from core.pen_tracker import PenTracker
from core.touch_calibration import (
    DEFAULT_SAMPLE_SECONDS,
    TouchCalibrationError,
    TouchSampleCollector,
    estimate_thresholds,
)
from ui.canvas import StrokeCanvas

__all__ = ["SessionDebug", "WhiteboardSession"]

_STATUS_COLORS = {
    "PEN DOWN": (0, 200, 0),
    "PEN UP": (200, 200, 200),
    "ERASE": (0, 215, 255),
    "HAND LOST": (0, 100, 255),
}


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
    pipeline_status: str = _PIPELINE_IDLE  # OCR/LLM 파이프라인 상태 (IDLE|RUNNING|DONE|ERROR)
    ocr_text: str | None = None
    corrected_text: str | None = None
    pipeline_error: str | None = None
    # 3D 디버그 패널(controller/overlay.py의 draw_3d_debug 우상단 패널)과 동일한 정보를
    # PyQt 쪽에서 Qt 라벨로 따로 그릴 수 있도록 노출한다 (영상 위에 텍스트를 굽지 않기 위함).
    plane_xy_mm: tuple[float, float] | None = None  # 손끝을 평면에 정사영한 좌표 (이번 프레임 성공 시)
    plane_size_mm: tuple[float, float] | None = None  # 캘리브레이션한 평면의 실제 크기(가로,세로 mm)
    contact_inside_plane: bool | None = None  # plane_xy_mm이 평면 범위 안인지 (3D 기준)
    contact_reprojection_error_px: float | None = None  # 이번 프레임 손 자세 재투영 오차(성공 시)
    contact_last_error_px: float | None = None  # 마지막 시도의 재투영 오차 (실패한 프레임 포함)


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
        calibration_path: str | Path | None = DEFAULT_CALIBRATION_PATH,
        touch_sample_seconds: float = DEFAULT_SAMPLE_SECONDS,
        intrinsics_path: str | Path | None = DEFAULT_INTRINSICS_PATH,
        plane_size_mm: tuple[float, float] = DEFAULT_PLANE_SIZE_MM,
        frame_size: tuple[int, int] = (1280, 720),
        use_3d_contact: bool = True,
        render_3d_debug_inline: bool = True,
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
        self._pipeline = OcrLlmPipelineWorker()
        self._last_pipeline_result = PipelineResult(_PIPELINE_IDLE, None, None, None)
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
        # 수치 패널(텍스트)을 프레임 픽셀에 직접 구울지 여부. OpenCV 데모는 별도 표시
        # 수단이 없어 그대로 굽고(True), PyQt 앱은 우측 패널의 Qt 라벨로 따로 보여주므로
        # False로 꺼서 영상 위에 큰 텍스트 박스가 덮이지 않게 한다. 수선 벡터·높이 눈금
        # (그래픽 오버레이)은 계속 프레임에 그린다 — 이건 "글자"가 아니라 시각적 표시라
        # Qt 라벨로 대체하기 애매하고, 손끝 위치와 겹쳐 봐야 의미가 있다.
        self._render_3d_debug_inline = render_3d_debug_inline

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
        contact_debug = frame.contact_debug if frame is not None else None
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
            pipeline_status=self._last_pipeline_result.status,
            ocr_text=self._last_pipeline_result.ocr_text,
            corrected_text=self._last_pipeline_result.corrected_text,
            pipeline_error=self._last_pipeline_result.error,
            plane_xy_mm=contact_debug.plane_xy_mm if contact_debug is not None else None,
            plane_size_mm=self._plane_size_mm if self._debug_3d else None,
            contact_inside_plane=contact_debug.inside_plane if contact_debug is not None else None,
            contact_reprojection_error_px=(
                contact_debug.reprojection_error_px if contact_debug is not None else None
            ),
            contact_last_error_px=(
                estimator.last_reprojection_error_px if estimator is not None else None
            ),
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
            self._calibration_message = err or "Place all 4 points first."
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
        mode_3d = "3D contact ON" if self._tracker.has_3d_contact else "3D contact OFF (pen_ratio fallback)"
        self._calibration_message = f"Calibration applied (dst={cal.dst_size}, {mode_3d})"
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
        # 화면 오버레이에 그대로 표시되므로 ASCII로만 작성 (Hershey 폰트에 한글 글리프 없음).
        return (
            f"3D: zero={zero_offset:.1f}mm thresh={result.down_thresh - zero_offset:.1f}/"
            f"{result.up_thresh - zero_offset:.1f}mm sep={separation:.1f}mm noise-sigma={noise:.1f}mm"
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
            failures.append("3D height samples don't separate contact/hover")

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
                f"pen_ratio thresh: down={result.down_thresh:.3f} up={result.up_thresh:.3f}"
            )

        self._touch_calibrating = False
        self._touch_phase = None
        self._touch_phase_started_at = None
        self._state_machine.reset()

        if parts:
            self._touch_message = " | ".join(parts)
            self._touch_message_ok = True
        else:
            self._touch_message = " / ".join(failures) or "Calibration failed."
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

    def trigger_ocr(self) -> bool:
        """현재 캔버스를 스냅샷해 OCR/LLM 파이프라인에 비동기로 넘긴다.

        캔버스가 아직 없거나(첫 프레임 이전) 이미 처리 중이면 아무것도 하지 않고
        ``False``를 반환한다. ``StrokeCanvas.image``는 라이브 버퍼 레퍼런스이므로
        ``.copy()``한 뒤 넘겨야 이후 그리기와 경합하지 않는다.
        """
        if self._canvas is None:
            return False
        return self._pipeline.submit(self._canvas.image.copy())

    def process_frame(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Run tracking + stroke recording and return the annotated frame."""
        # 모드(캘리브레이션 포함)와 무관하게 매 프레임 소비한다 — 완료된 작업이 다음
        # trigger_ocr() 호출에 유실되지 않도록, 그리고 캘리브레이션 중에도 백그라운드
        # 결과가 계속 반영되도록.
        self._last_pipeline_result = self._pipeline.poll()

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
            return overlay.draw_calibration_overlay(
                bgr_frame,
                picker=self._calibration_picker,
                fingertip=self._calibration_fingertip,
                message=self.calibration_message,
            )

        if self._touch_calibrating:
            # 터치 캘리브레이션 중에도 캔버스에는 그리지 않고 pen_ratio 표본만 모은다.
            touch_result = self._tracker.process(bgr_frame)
            self._advance_touch_calibration(touch_result.pen_ratio, touch_result.raw_height_mm)
            return overlay.draw_touch_calibration_overlay(
                bgr_frame,
                phase=self._touch_phase,
                progress=self.touch_progress,
                fingertip=touch_result.fingertip,
                message=self.touch_message,
            )

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

        annotated = self._canvas.overlay(bgr_frame)
        if self._calibration is not None:
            # 캘리브레이션한 필기 영역 경계를 항상 얇게 겹쳐 그려, 게이팅 기준을
            # 눈으로 확인할 수 있게 한다.
            outline = self._calibration.src_points.as_array().astype(np.int32)
            cv2.polylines(annotated, [outline], True, overlay.PLANE_OUTLINE_COLOR, 1, cv2.LINE_AA)
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
                overlay.CAL_ERR_COLOR,
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
                overlay.CAL_OK_COLOR if result.pen_down else overlay.CAL_TEXT_COLOR,
                2,
                cv2.LINE_AA,
            )
        if self._debug_3d:
            overlay.draw_3d_debug(
                annotated, result, tracker=self._tracker, plane_size_mm=self._plane_size_mm,
                show_panel=self._render_3d_debug_inline,
            )
        return annotated

    def _draw_to(self, fingertip: tuple[int, int]) -> None:
        """펜을 내린 상태로 손끝 위치까지 획을 잇는다."""
        if self._canvas is None:
            return
        if self._canvas.is_pen_down:
            self._canvas.move(fingertip)
        else:
            self._canvas.pen_down(fingertip)

    def close(self) -> None:
        self._pipeline.shutdown(wait=False)
        self._tracker.close()

    def __enter__(self) -> "WhiteboardSession":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
