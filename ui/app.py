"""
화이트보드 세션용 PyQt 레이아웃 골격 (2주차 범위).

`controller/main.py``와 동일한 파이프라인을 Qt 창 안에서 실행합니다. 버튼/단축키로
자동·수동 모드 전환(M)과 좌표 CSV 기록(R)을 제어한다.
우측 열에는 캔버스 소형 미리보기와 pen_ratio 디버그 패널(RatioBar)을 표시한다.

5주차 스캐폴딩에서 QThread 워커를 추가할 예정입니다. 그때까지는 이 파일을 렌더링 전용으로 사용합니다(llm/ocr 호출x)

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
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from controller.main import SessionDebug, WhiteboardSession

_FRAME_INTERVAL_MS = 33  # ~30 fps

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

    def __init__(self, camera_index: int, *, mirror: bool = False) -> None:
        super().__init__()
        self.setWindowTitle("Virtual Whiteboard")

        self._camera = cv2.VideoCapture(camera_index)
        if not self._camera.isOpened():
            raise RuntimeError(f"카메라 {camera_index}을(를) 열 수 없습니다.")
        self._mirror = mirror
        self._session = WhiteboardSession()

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

        right_panel = QWidget()
        right_panel.setFixedWidth(300)
        right_layout = QVBoxLayout()
        right_layout.addWidget(canvas_title)
        right_layout.addWidget(self._canvas_view)
        right_layout.addWidget(self._ratio_bar)
        right_layout.addWidget(self._status_label1)
        right_layout.addWidget(self._status_label2)
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

        self._record_button = QPushButton()
        self._record_button.setCheckable(True)
        self._record_button.toggled.connect(self._on_record_toggled)

        clear_button = QPushButton("지우기")
        clear_button.clicked.connect(self._session.clear)
        save_button = QPushButton("저장")
        save_button.clicked.connect(self._save_canvas)
        quit_button = QPushButton("종료")
        quit_button.clicked.connect(self.close)

        self._refresh_mode_button()
        self._refresh_record_button()

        buttons = QHBoxLayout()
        for button in (
            self._mode_button,
            self._pen_button,
            self._record_button,
            clear_button,
            save_button,
            quit_button,
        ):
            buttons.addWidget(button)

        top_layout = QHBoxLayout()
        top_layout.addWidget(self._video, stretch=1)
        top_layout.addWidget(self._right_panel, stretch=0)

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

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if event.key() == Qt.Key.Key_Space:
            # 자동 모드에서는 펜 버튼이 비활성화되어 있으므로 토글하지 않는다
            # (비활성 버튼도 프로그램 호출로는 토글되어 수동 복귀 시 상태가 어긋난다).
            if self._pen_button.isEnabled():
                self._pen_button.toggle()
        elif event.key() == Qt.Key.Key_M:
            self._mode_button.toggle()
        elif event.key() == Qt.Key.Key_R:
            self._record_button.toggle()
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

    def _on_record_toggled(self, on: bool) -> None:
        if on:
            path = self._session.start_recording()
            self.statusBar().showMessage(f"CSV 기록 시작: {path}")
        else:
            self._session.stop_recording()
            self.statusBar().showMessage("CSV 기록 중지")
        self._refresh_record_button()

    def _refresh_record_button(self) -> None:
        self._record_button.setText("기록 중지 (R)" if self._session.is_recording else "기록 (R)")

    def _update_frame(self) -> None:
        ok, frame = self._camera.read()
        if not ok:
            self.statusBar().showMessage("카메라 프레임을 읽지 못했습니다.")
            return
        if self._mirror:
            frame = cv2.flip(frame, 1)

        annotated = self._session.process_frame(frame)
        status = f"[{self._session.mode_name}] {self._session.status}"
        if self._session.is_recording:
            status += "  ● REC"
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
            self._status_label1.setStyleSheet("color: #1a8a3c;")
        elif debug.status in ("HAND LOST", "ERASE"):
            self._status_label1.setStyleSheet("color: #cc7a00;")
        else:
            self._status_label1.setStyleSheet("color: #333333;")

        if not debug.hand_detected:
            self._status_label2.setText("손 미검출")
            self._status_label2.setStyleSheet("color: #cc7a00;")
        else:
            instant = "DOWN" if debug.instant_pen_down else "UP"
            stable = "DOWN" if debug.stable_pen_down else "UP"
            self._status_label2.setText(f"순간: {instant} / 안정: {stable}")
            self._status_label2.setStyleSheet(
                "color: #1a8a3c;" if debug.stable_pen_down else "color: #555555;"
            )

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
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = WhiteboardWindow(args.camera, mirror=args.mirror)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
