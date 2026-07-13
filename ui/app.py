"""
화이트보드 세션용 PyQt 레이아웃 골격 (2주차 범위).

`controller/main.py``와 동일한 파이프라인을 Qt 창 안에서 실행합니다. 버튼/단축키로
자동·수동 모드 전환(M)과 좌표 CSV 기록(R)을 제어한다.

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
from PyQt6.QtGui import QCloseEvent, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from controller.main import WhiteboardSession

_FRAME_INTERVAL_MS = 33  # ~30 fps


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

        layout = QVBoxLayout()
        layout.addWidget(self._video, stretch=1)
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
