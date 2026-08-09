"""
화이트보드 세션용 PyQt 레이아웃 골격 (2주차 범위).

`controller/main.py``와 동일한 파이프라인을 Qt 창 안에서 실행합니다. 버튼/단축키로
자동·수동 모드 전환(M), OCR 트리거(O), 4점 캘리브레이션(K)·접촉 캘리브레이션(T)을 제어한다.
우측 열에는 캔버스 소형 미리보기와 pen_ratio 디버그 패널(RatioBar)을 표시한다.

OCR/LLM 백그라운드 처리는 `controller.ocr_llm_pipeline.OcrLlmPipelineWorker`
(ThreadPoolExecutor 기반)가 담당한다. `WhiteboardSession`은 OpenCV 데모와 공유되는
UI-독립 엔트리포인트라 QThread 대신 스레드 세이프 폴링 방식을 쓴다 — 워커 스레드는
Qt 위젯을 건드리지 않고, `_update_frame()`이 GUI 스레드에서 `session.debug`(plain
dataclass)만 매 틱 읽는다.

실행 명령어: `python -m ui.app`
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QCloseEvent, QColor, QImage, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from controller.session import SessionDebug, WhiteboardSession
from core.contact3d import MAX_REPROJECTION_ERROR_PX

_FRAME_INTERVAL_MS = 33  # ~30 fps

# 전체화면/큰 창에서도 읽히도록 우측 패널 글자 크기를 기본 Qt 폰트보다 키운다.
_LABEL_FONT_PX = 14
_DEBUG3D_FONT_PX = 13


def _status_style(color: str) -> str:
    """상태 라벨(상태/손/OCR/캘리브레이션)용 스타일 — 항상 폰트 크기를 포함한다.

    라벨마다 색만 다르게 여러 곳(_update_debug_labels 등)에서 setStyleSheet를 새로
    호출하는데, 매번 이 헬퍼를 거치게 해서 font-size가 누락되는 일을 막는다.
    """
    return f"font-size: {_LABEL_FONT_PX}px; color: {color};"


def _debug3d_style(color: str) -> str:
    """3D 디버그 카드 라벨용 스타일 — 배경/테두리를 포함해 위 라벨들과 구분되어 보이게 한다."""
    return (
        f"font-family: Consolas, 'DejaVu Sans Mono', monospace; font-size: {_DEBUG3D_FONT_PX}px;"
        f" color: {color}; background-color: #f2f2f2; border: 1px solid #ddd;"
        " border-radius: 4px; padding: 6px;"
    )

# RatioBar가 pen_ratio를 x 픽셀로 매핑할 때 쓰는 기본 표시 범위.
_RATIO_RANGE_LO = 0.0
_RATIO_RANGE_HI = 1.2


def ratio_to_x(ratio: float, width: int, lo: float = _RATIO_RANGE_LO, hi: float = _RATIO_RANGE_HI) -> int:
    """``ratio``를 [lo, hi] 구간 기준 [0, width] 픽셀 x좌표로 변환한다 (클램프 포함).

    ``lo >= hi``인 퇴화 구간에서는 0을 반환한다 (0으로 나누기 방어).
    """
    if width <= 0 or hi <= lo:
        return 0
    clamped = max(lo, min(hi, ratio))
    fraction = (clamped - lo) / (hi - lo)
    return int(round(fraction * width))


class RatioBar(QWidget):
    """pen_ratio 값, down/up 임계선, 안정 판정 색상, 지우기 진행 게이지를 그리는 디버그 바."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(56)
        self.setMinimumWidth(260)
        self._debug: SessionDebug | None = None

    def set_debug(self, debug: SessionDebug) -> None:
        self._debug = debug
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        width, height = self.width(), self.height()
        painter.fillRect(0, 0, width, height, QColor(235, 235, 235))

        debug = self._debug
        if debug is None or debug.pen_ratio is None:
            painter.setPen(QColor(130, 130, 130))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "NO HAND")
            painter.end()
            return

        bar_top = 16
        bar_bottom = height - 10
        bar_height = bar_bottom - bar_top

        fill_x = ratio_to_x(debug.pen_ratio, width)
        fill_color = QColor(40, 170, 70) if debug.stable_pen_down else QColor(150, 150, 150)
        painter.fillRect(0, bar_top, fill_x, bar_height, fill_color)

        down_x = ratio_to_x(debug.down_thresh, width)
        painter.setPen(QColor(200, 30, 30))
        painter.drawLine(down_x, bar_top, down_x, bar_bottom)
        painter.drawText(max(down_x - 14, 0), bar_top - 4, f"{debug.down_thresh:.2f}")

        up_x = ratio_to_x(debug.up_thresh, width)
        painter.setPen(QColor(20, 90, 200))
        painter.drawLine(up_x, bar_top, up_x, bar_bottom)
        painter.drawText(min(up_x + 2, max(width - 30, 0)), bar_top - 4, f"{debug.up_thresh:.2f}")

        if debug.erase_progress > 0.0:
            gauge_w = ratio_to_x(debug.erase_progress, width, 0.0, 1.0)
            painter.fillRect(0, bar_bottom, gauge_w, height - bar_bottom, QColor(255, 140, 0))

        painter.setPen(QColor(50, 50, 50))
        painter.drawText(4, height - 2, f"ratio={debug.pen_ratio:.2f}")
        painter.end()


class WhiteboardWindow(QMainWindow):

    def __init__(self, camera_index: int, *, mirror: bool = False, flip_vertical: bool = False) -> None:
        super().__init__()
        self.setWindowTitle("Virtual Whiteboard")

        self._camera = cv2.VideoCapture(camera_index)
        if not self._camera.isOpened():
            raise RuntimeError(f"카메라 {camera_index}을(를) 열 수 없습니다.")
        self._mirror = mirror
        self._flip_vertical = flip_vertical
        # 3D 디버그 수치 패널은 영상 위에 굽지 않고 우측 Qt 라벨(self._debug3d_label)로
        # 따로 보여준다 — 화면이 커지면 텍스트 박스가 실제 손 위치를 덮어버리는 문제가 있었음.
        self._session = WhiteboardSession(render_3d_debug_inline=False)

        self._video = QLabel("카메라 대기 중...")
        self._video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video.setMinimumSize(640, 480)

        canvas_title = QLabel("캔버스")
        self._canvas_view = QLabel("캔버스 대기 중...")
        self._canvas_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._canvas_view.setMinimumHeight(180)
        self._canvas_view.setStyleSheet("border: 1px solid #bbb; background-color: white;")

        self._ratio_bar = RatioBar()
        self._status_label1 = QLabel("상태: -")
        self._status_label2 = QLabel("손 미검출")
        self._pipeline_label = QLabel("OCR/LLM: 대기")
        self._calibration_label = QLabel("")
        for label in (self._status_label1, self._status_label2, self._pipeline_label,
                      self._calibration_label):
            label.setWordWrap(True)
            label.setStyleSheet(_status_style("#333333"))

        # 3D 디버그 수치 패널 — 영상 위에 굽는 대신 여기 모노스페이스 라벨로 표시한다
        # (3D 디버그가 꺼져 있으면 빈 문자열, _update_debug3d_label()이 채운다). 위쪽
        # 라벨들과 붙어서 한 덩어리로 보이지 않도록 구분선 + 카드형 배경을 둔다.
        self._debug3d_separator = QFrame()
        self._debug3d_separator.setFrameShape(QFrame.Shape.HLine)
        self._debug3d_separator.setStyleSheet("color: #ccc; margin-top: 6px;")
        self._debug3d_label = QLabel("")
        self._debug3d_label.setWordWrap(True)
        self._debug3d_label.setStyleSheet(_debug3d_style("#333333"))

        right_panel = QWidget()
        right_panel.setMinimumWidth(320)
        right_layout = QVBoxLayout()
        right_layout.addWidget(canvas_title)
        right_layout.addWidget(self._canvas_view)
        right_layout.addWidget(self._ratio_bar)
        right_layout.addWidget(self._status_label1)
        right_layout.addWidget(self._status_label2)
        right_layout.addWidget(self._pipeline_label)
        right_layout.addWidget(self._calibration_label)
        right_layout.addWidget(self._debug3d_separator)
        right_layout.addWidget(self._debug3d_label)
        right_layout.addStretch(1)
        right_panel.setLayout(right_layout)
        self._right_panel = right_panel

        self._mode_button = QPushButton()
        self._mode_button.setCheckable(True)
        self._mode_button.setChecked(self._session.auto_mode)
        self._mode_button.toggled.connect(self._on_mode_toggled)

        self._pen_button = QPushButton("펜 (SPACE)")
        self._pen_button.setCheckable(True)
        self._pen_button.setEnabled(not self._session.auto_mode)  # 수동 모드에서만
        self._pen_button.toggled.connect(self._session.set_pen_down)

        self._ocr_button = QPushButton("OCR 트리거 (O)")
        self._ocr_button.clicked.connect(self._on_ocr_clicked)

        self._cal_button = QPushButton("캘리브레이션 (K)")
        self._cal_button.clicked.connect(self._on_calibrate_clicked)

        self._touch_cal_button = QPushButton("접촉 캘리브레이션 (T)")
        self._touch_cal_button.clicked.connect(self._on_touch_calibrate_clicked)

        # 3D 접촉 판정 근거: 평면 수선 벡터·높이 게이지(그래픽)는 영상 위에 겹쳐 그리고,
        # 수치 요약(우상단 패널에 해당하던 내용)은 우측 self._debug3d_label로 따로 본다.
        self._debug3d_button = QPushButton("3D 디버그 (D)")
        self._debug3d_button.setCheckable(True)
        self._debug3d_button.setChecked(self._session.debug_3d)
        self._debug3d_button.toggled.connect(self._session.set_debug_3d)

        self._clear_button = QPushButton("지우기")
        self._clear_button.clicked.connect(self._session.clear)
        self._save_button = QPushButton("저장")
        self._save_button.clicked.connect(self._save_canvas)
        quit_button = QPushButton("종료")
        quit_button.clicked.connect(self.close)

        self._refresh_mode_button()

        buttons = QHBoxLayout()
        for button in (
            self._mode_button,
            self._pen_button,
            self._ocr_button,
            self._cal_button,
            self._touch_cal_button,
            self._debug3d_button,
            self._clear_button,
            self._save_button,
            quit_button,
        ):
            # QPushButton은 기본적으로 키보드 포커스를 받고, 포커스가 있는 상태에서
            # SPACE/Enter를 누르면 keyPressEvent()에 도달하기 전에 버튼 자신의
            # click()을 먼저 실행해버린다 (예: 종료 버튼이 포커스를 가지면 SPACE로
            # 캘리브레이션 점을 찍으려다 앱이 종료됨). 단축키는 keyPressEvent()에서
            # 중앙집중식으로 처리하므로 버튼은 마우스 클릭 전용으로 고정한다.
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            buttons.addWidget(button)

        top_layout = QHBoxLayout()
        top_layout.addWidget(self._video, stretch=3)
        top_layout.addWidget(self._right_panel, stretch=1)

        layout = QVBoxLayout()
        layout.addLayout(top_layout, stretch=1)
        layout.addLayout(buttons)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.statusBar().showMessage("PEN UP")

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_frame)
        self._timer.start(_FRAME_INTERVAL_MS)

        if self._session.needs_calibration:
            # 세션 시작 시 유효한 캘리브레이션이 없으면(최초 실행) 1회 강제한다 —
            # OpenCV 데모(controller/main.py)의 동일 로직과 맞춘다.
            self._session.start_calibration()
            self.statusBar().showMessage(
                "저장된 캘리브레이션이 없어 최초 지정을 시작합니다 — "
                "손끝을 모서리에 놓고 SPACE (TL→TR→BR→BL)"
            )
        self._refresh_calibration_controls()

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if self._session.is_calibrating:
            if event.key() == Qt.Key.Key_Z:
                self._session.undo_calibration_point()
            elif event.key() == Qt.Key.Key_X:
                self._session.reset_calibration_points()
            elif event.key() == Qt.Key.Key_Space:
                if not self._session.add_calibration_point_at_fingertip():
                    self.statusBar().showMessage("손이 검출되지 않아 점을 찍을 수 없습니다.")
            elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_S):
                self._session.confirm_calibration()
                self._refresh_calibration_controls()
            elif event.key() == Qt.Key.Key_Escape and self._session.has_calibration:
                self._session.cancel_calibration()
                self._refresh_calibration_controls()
            return
        if self._session.is_touch_calibrating:
            if event.key() == Qt.Key.Key_Escape:
                self._session.cancel_touch_calibration()
                self._refresh_calibration_controls()
            return
        if event.key() == Qt.Key.Key_Space:
            # 자동 모드에서는 펜 버튼이 비활성화되어 있으므로 토글하지 않는다
            # (비활성 버튼도 프로그램 호출로는 토글되어 수동 복귀 시 상태가 어긋난다).
            if self._pen_button.isEnabled():
                self._pen_button.toggle()
        elif event.key() == Qt.Key.Key_M:
            self._mode_button.toggle()
        elif event.key() == Qt.Key.Key_O:
            self._on_ocr_clicked()
        elif event.key() == Qt.Key.Key_D:
            self._debug3d_button.toggle()
        elif event.key() == Qt.Key.Key_K:
            self._on_calibrate_clicked()
        elif event.key() == Qt.Key.Key_T:
            self._on_touch_calibrate_clicked()
        else:
            super().keyPressEvent(event)

    def _on_mode_toggled(self, auto: bool) -> None:
        self._session.set_auto_mode(auto)
        self._refresh_mode_button()
        # 자동 모드에서는 수동 펜 버튼을 잠근다.
        self._pen_button.setEnabled(not auto)
        if auto:
            self._pen_button.setChecked(False)

    def _refresh_mode_button(self) -> None:
        self._mode_button.setText(f"모드: {self._session.mode_name} (M)")

    def _on_ocr_clicked(self) -> None:
        if not self._session.trigger_ocr():
            self.statusBar().showMessage("이전 작업 처리 중 — 잠시 후 다시 시도하세요.")

    def _on_calibrate_clicked(self) -> None:
        self._session.start_calibration()
        self._refresh_calibration_controls()

    def _on_touch_calibrate_clicked(self) -> None:
        self._session.start_touch_calibration()
        self._refresh_calibration_controls()

    def _refresh_calibration_controls(self) -> None:
        """캘리브레이션/터치캘리브레이션 진행 중이면 나머지 컨트롤을 잠근다.

        OpenCV 데모의 배타적 key 분기(캘리브레이션 중엔 다른 키가 먹히지 않음)와
        동일한 효과를 버튼에도 적용한다.
        """
        busy = self._session.is_calibrating or self._session.is_touch_calibrating
        self._cal_button.setEnabled(not busy)
        self._touch_cal_button.setEnabled(not busy)
        self._mode_button.setEnabled(not busy)
        self._ocr_button.setEnabled(not busy)
        self._debug3d_button.setEnabled(not busy)
        self._clear_button.setEnabled(not busy)
        self._save_button.setEnabled(not busy)
        self._pen_button.setEnabled(not busy and not self._session.auto_mode)

    def _update_frame(self) -> None:
        ok, frame = self._camera.read()
        if not ok:
            self.statusBar().showMessage("카메라 프레임을 읽지 못했습니다.")
            return
        if self._mirror:
            frame = cv2.flip(frame, 1)
        if self._flip_vertical:
            frame = cv2.flip(frame, 0)

        annotated = self._session.process_frame(frame)
        status = f"[{self._session.mode_name}] {self._session.status}"
        self.statusBar().showMessage(status)

        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        height, width, _ = rgb.shape
        image = QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888)
        self._video.setPixmap(
            QPixmap.fromImage(image).scaled(
                self._video.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

        self._update_canvas_view()
        debug = self._session.debug
        self._ratio_bar.set_debug(debug)
        self._update_debug_labels(debug)
        self._update_pipeline_label(debug)
        self._update_calibration_label()
        self._update_debug3d_label(debug)

    def _update_canvas_view(self) -> None:
        canvas = self._session.canvas
        if canvas is None:
            self._canvas_view.setText("캔버스 대기 중...")
            return
        canvas_rgb = cv2.cvtColor(canvas.image, cv2.COLOR_BGR2RGB)
        c_height, c_width, _ = canvas_rgb.shape
        canvas_image = QImage(
            canvas_rgb.data, c_width, c_height, 3 * c_width, QImage.Format.Format_RGB888
        )
        self._canvas_view.setPixmap(
            QPixmap.fromImage(canvas_image).scaled(
                self._canvas_view.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _update_debug_labels(self, debug: SessionDebug) -> None:
        status_text = f"상태: {debug.status}"
        if debug.pending:
            status_text += f" ({debug.pending})"
        self._status_label1.setText(status_text)
        if debug.status == "PEN DOWN":
            self._status_label1.setStyleSheet(_status_style("#1a8a3c"))
        elif debug.status in ("HAND LOST", "ERASE"):
            self._status_label1.setStyleSheet(_status_style("#cc7a00"))
        else:
            self._status_label1.setStyleSheet(_status_style("#333333"))

        if not debug.hand_detected:
            self._status_label2.setText("손 미검출")
            self._status_label2.setStyleSheet(_status_style("#cc7a00"))
        else:
            instant = "DOWN" if debug.instant_pen_down else "UP"
            stable = "DOWN" if debug.stable_pen_down else "UP"
            text = f"순간: {instant} / 안정: {stable}"
            if debug.height_mm is not None:
                # 3D 판정 중이면 판정에 실제로 쓰인 높이를 함께 보여준다 (Qt는 한글 표시 가능).
                text += f"  |  높이 {debug.height_mm:+.1f}mm"
            self._status_label2.setText(text)
            self._status_label2.setStyleSheet(
                _status_style("#1a8a3c" if debug.stable_pen_down else "#555555")
            )

    def _update_pipeline_label(self, debug: SessionDebug) -> None:
        if debug.pipeline_status == "RUNNING":
            self._pipeline_label.setText("OCR/LLM: 처리 중...")
            self._pipeline_label.setStyleSheet(_status_style("#cc7a00"))
        elif debug.pipeline_status == "DONE":
            self._pipeline_label.setText(f"OCR/LLM: 완료 — {debug.corrected_text}")
            self._pipeline_label.setStyleSheet(_status_style("#1a8a3c"))
        elif debug.pipeline_status == "ERROR":
            self._pipeline_label.setText(f"OCR/LLM: 오류 — {debug.pipeline_error}")
            self._pipeline_label.setStyleSheet(_status_style("#cc2222"))
        else:
            self._pipeline_label.setText("OCR/LLM: 대기")
            self._pipeline_label.setStyleSheet(_status_style("#333333"))

    def _update_calibration_label(self) -> None:
        if self._session.is_calibrating:
            picker = self._session.calibration_picker
            if picker is not None and not picker.is_complete:
                text = f"[캘리브레이션] 다음: {picker.next_label} ({picker.count}/4) — 손끝+SPACE"
            else:
                text = "[캘리브레이션] 4점 완료 — Enter/S 적용 · Z 취소 · X 리셋 · Esc 취소"
            color = "#333333"
        elif self._session.is_touch_calibrating:
            phase_label = "1/2 hover (띄우기)" if self._session.touch_phase == "hover" else "2/2 touch (누르기)"
            text = f"[접촉 캘리브레이션] {phase_label} — {self._session.touch_progress * 100:.0f}% (Esc 취소)"
            color = "#333333"
        else:
            msg = self._session.calibration_message or self._session.touch_message
            if msg is not None:
                text, ok = msg
                color = "#1a8a3c" if ok else "#cc2222"
            else:
                text = "캘리브레이션: 저장됨" if self._session.has_calibration else "캘리브레이션: 없음"
                color = "#333333"
        self._calibration_label.setText(text)
        self._calibration_label.setStyleSheet(_status_style(color))

    def _update_debug3d_label(self, debug: SessionDebug) -> None:
        """controller/overlay.py의 draw_3d_debug() 우상단 패널과 같은 정보를 Qt 라벨로.

        영상 위에 굽지 않고(WhiteboardSession(render_3d_debug_inline=False)) 여기서
        그리는 이유는 화면을 키우면 그 텍스트 박스가 실제 손 위치를 덮어버렸기 때문.
        """
        if not debug.debug_3d:
            self._debug3d_label.setText("")
            return
        if debug.height_mm is None or debug.plane_xy_mm is None:
            # draw_3d_debug()의 "OFF/실패 사유" 분기와 동일한 이유를 보여준다.
            if not debug.has_3d_contact:
                reason = "3D DEBUG: OFF\n(4점 캘리브레이션 + 평면 크기 + intrinsics 필요, K)"
            elif debug.contact_last_error_px is not None:
                reason = (
                    f"3D DEBUG: 손 재투영 오차 {debug.contact_last_error_px:.0f}px"
                    f" > {MAX_REPROJECTION_ERROR_PX:.0f}px 문턱\n(pen_ratio로 폴백)"
                )
            else:
                reason = "3D DEBUG: 이번 프레임 자세 추정 실패\n(pen_ratio로 폴백)"
            self._debug3d_label.setText(reason)
            self._debug3d_label.setStyleSheet(_debug3d_style("#cc2222"))
            return

        plane_w, plane_h = debug.plane_size_mm or (0.0, 0.0)
        plane_x, plane_y = debug.plane_xy_mm
        inside = debug.contact_inside_plane
        zero_text = f"{debug.zero_offset_mm:+.1f} mm" if debug.zero_offset_mm is not None else "n/a (T)"
        lines = [
            "3D DEBUG",
            f"height     {debug.height_mm:+.1f} mm",
            f"raw height {debug.raw_height_mm:+.1f} mm" if debug.raw_height_mm is not None else "raw height n/a",
            f"zero offs  {zero_text}",
            (
                f"thresh d/u {debug.contact_down_mm:.1f} / {debug.contact_up_mm:.1f} mm"
                if debug.contact_down_mm is not None and debug.contact_up_mm is not None
                else "thresh d/u n/a"
            ),
            f"plane XY   {plane_x:.0f}, {plane_y:.0f} of {plane_w:.0f}x{plane_h:.0f}"
            + ("" if inside else "  OUT"),
            (
                f"hand reproj {debug.contact_reprojection_error_px:.1f} px"
                if debug.contact_reprojection_error_px is not None
                else "hand reproj n/a"
            ),
            f"source {debug.contact_source}   contact {'YES' if debug.instant_pen_down else 'no'}",
        ]
        self._debug3d_label.setText("\n".join(lines))
        color = "#cc2222" if inside is False else "#333333"
        self._debug3d_label.setStyleSheet(_debug3d_style(color))

    def _save_canvas(self) -> None:
        saved = self._session.save_canvas(
            Path("captures") / time.strftime("canvas_%Y%m%d_%H%M%S.png")
        )
        message = f"캔버스 저장: {saved}" if saved else "저장할 캔버스가 없습니다."
        self.statusBar().showMessage(message)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 (Qt naming)
        self._timer.stop()
        self._camera.release()
        self._session.close()
        super().closeEvent(event)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="camera device index")
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="flip the frame horizontally (selfie view; off by default for the desk view)",
    )
    parser.add_argument(
        "--flip-vertical",
        action="store_true",
        help="flip the frame vertically (oblique webcam upside-down fix; independent of --mirror)",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = WhiteboardWindow(args.camera, mirror=args.mirror, flip_vertical=args.flip_vertical)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
