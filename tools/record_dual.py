"""
한 창에서 메인(사선) 카메라와 측면(정답 판정용) 카메라를 동시에 모니터링하고,
촬영자/조명/속도 조건을 입력받아 규칙에 맞는 파일명으로 동시 녹화한다.

저장물 (data/dataset/recordings/):
  이름_조명_속도_회차_main.mp4   # 사선 뷰 — 이후 랜드마크(feature) 추출용
  이름_조명_속도_회차_side.mp4   # 측면 뷰 — 접촉 여부 정답(라벨) 판정용
  이름_조명_속도_회차_frames.csv # 프레임별 타임스탬프 + main/side 프레임 수신 여부(0/1)
                                 # (두 영상 싱크 및 프레임 누락 검출 근거)

랜드마크 추출은 녹화 중 하지 않는다(프레임 드랍 방지) — 녹화 후 main 영상에서
오프라인으로 추출한다.

측면 카메라로는 스마트폰을 USB 웹캠(화면 미러링 앱 등)으로 연결해 사용하는 것을
전제로 한다. "카메라 검색" 버튼을 누르면 인덱스 0~7을 순회하며 열리는 카메라와
해상도를 상태바에 보여준다.

"검출 테스트" 버튼을 켜면 메인 프리뷰에 MediaPipe 손 검출 결과(랜드마크 + HAND OK/
NO HAND)를 겹쳐 보여준다 — 녹화 전에 현재 각도에서 손이 잡히는지 점검하는 용도이며,
녹화를 시작하면 자동으로 꺼진다(프레임 드랍 방지, 녹화 영상에는 표시가 남지 않음).

실행: python -m tools.record_dual
"""

from __future__ import annotations

import csv
import re
import sys
import time
from pathlib import Path

import cv2
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QCloseEvent, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

OUTPUT_DIR = Path("data/dataset/recordings")
_FRAME_INTERVAL_MS = 33  # ~30 fps
_NOMINAL_FPS = 30.0

# 카메라 검색이 빈 인덱스를 프로브할 때 OpenCV가 콘솔에 경고/에러를 대량으로 뿌린다
# (DSHOW "can't be used to capture by index", obsensor "index out of range" 등).
# 기능과 무관한 소음이라 묵음 처리한다.
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except AttributeError:  # 구버전 opencv에는 utils.logging이 없다
    pass


def build_basename(name: str, light: str, speed: str, take: int) -> str:
    return f"{name}_{light}_{speed}_{take:02d}"


def next_take(output_dir: Path, name: str, light: str, speed: str) -> int:
    """같은 조건으로 이미 찍은 파일이 있으면 다음 회차 번호를 돌려준다."""
    pattern = re.compile(re.escape(f"{name}_{light}_{speed}_") + r"(\d+)_main\.mp4$")
    takes = [
        int(m.group(1))
        for p in output_dir.glob("*_main.mp4")
        if (m := pattern.match(p.name))
    ]
    return max(takes, default=0) + 1


def _open_capture(index: int) -> cv2.VideoCapture:
    """지정한 인덱스로 카메라를 연다.

    Windows에서는 기본 백엔드(MSMF)가 USB/가상 웹캠(스마트폰 미러링 앱 등)에서
    느리거나 아예 열리지 않는 경우가 많아 DSHOW 백엔드를 먼저 시도하고,
    실패하면 기본 백엔드로 폴백한다. 다른 플랫폼에서는 기본 백엔드만 사용한다.
    """
    if sys.platform == "win32":
        cam = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cam.isOpened():
            cam.release()
            # CAP_ANY 폴백은 obsensor 등 무관한 백엔드까지 순회하므로 MSMF로 한정한다.
            cam = cv2.VideoCapture(index, cv2.CAP_MSMF)
    else:
        cam = cv2.VideoCapture(index)
    if cam.isOpened():
        # 스마트폰 등 고해상도 카메라의 인코딩 부하를 줄이기 위한 요청값.
        # 카메라가 무시할 수 있으며 그래도 무방하다.
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cam


def to_pixmap(bgr_frame, target: QLabel) -> QPixmap:
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    height, width, _ = rgb.shape
    image = QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image).scaled(
        target.size(),
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


class RecorderWindow(QMainWindow):
    """조건 입력 폼 + 듀얼 프리뷰 + 녹화 제어를 한 창에 담은 수집 도구."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("데이터 수집 - 듀얼 카메라 녹화")

        self._cams: dict[str, cv2.VideoCapture | None] = {"main": None, "side": None}
        self._open_indices: set[int] = set()
        self._test_tracker = None  # "검출 테스트" 토글 시에만 지연 생성되는 HandTracker
        self._writers: dict[str, cv2.VideoWriter] = {}
        self._csv_file = None
        self._csv_writer = None
        self._recording = False
        self._record_start = 0.0
        self._frame_index = 0

        # --- 입력 폼 ---
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("예: minjin")
        self._light_box = QComboBox()
        self._light_box.addItems(["bright", "dark"])
        self._speed_box = QComboBox()
        self._speed_box.addItems(["slow", "fast"])
        self._main_index = QSpinBox()
        self._main_index.setValue(0)
        self._side_index = QSpinBox()
        self._side_index.setValue(1)
        open_button = QPushButton("카메라 열기")
        open_button.clicked.connect(self._open_cameras)
        scan_button = QPushButton("카메라 검색")
        scan_button.clicked.connect(self._scan_cameras)
        self._detect_button = QPushButton("검출 테스트")
        self._detect_button.setCheckable(True)
        self._detect_button.toggled.connect(self._on_detect_toggled)
        self._detect_button.setToolTip(
            "녹화 전 각도 점검용: 메인 프리뷰에 MediaPipe 손 검출 결과를 표시합니다.\n"
            "녹화를 시작하면 자동으로 꺼집니다 (프레임 드랍 방지)."
        )

        form = QGridLayout()
        for col, (label, widget) in enumerate(
            [
                ("촬영자", self._name_edit),
                ("조명", self._light_box),
                ("속도", self._speed_box),
                ("메인 카메라", self._main_index),
                ("측면 카메라", self._side_index),
            ]
        ):
            form.addWidget(QLabel(label), 0, col)
            form.addWidget(widget, 1, col)
        form.addWidget(open_button, 1, 5)
        form.addWidget(scan_button, 1, 6)
        form.addWidget(self._detect_button, 1, 7)

        # --- 프리뷰 ---
        self._previews: dict[str, QLabel] = {}
        previews = QHBoxLayout()
        for key, title in (("main", "메인 (사선 = 학습 입력)"), ("side", "측면 (= 정답 라벨용)")):
            box = QVBoxLayout()
            caption = QLabel(title)
            caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
            view = QLabel("카메라 대기 중")
            view.setAlignment(Qt.AlignmentFlag.AlignCenter)
            view.setMinimumSize(480, 360)
            box.addWidget(caption)
            box.addWidget(view, stretch=1)
            previews.addLayout(box)
            self._previews[key] = view

        # --- 녹화 제어 ---
        self._record_button = QPushButton("● 녹화 시작")
        self._record_button.setEnabled(False)
        self._record_button.clicked.connect(self._toggle_recording)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addLayout(previews, stretch=1)
        layout.addWidget(self._record_button)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.statusBar().showMessage("조건을 입력하고 '카메라 열기'를 누르세요")

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_frames)
        self._timer.start(_FRAME_INTERVAL_MS)

    # --- 카메라 ---

    def _open_cameras(self) -> None:
        if self._recording:
            return
        self._release_cameras()
        indices = {"main": self._main_index.value(), "side": self._side_index.value()}
        opened = []
        for key, index in indices.items():
            cam = _open_capture(index)
            if cam.isOpened():
                self._cams[key] = cam
                opened.append(key)
            else:
                cam.release()
                self._previews[key].setText(f"카메라 {index}을(를) 열 수 없음")
        self._open_indices = {indices[key] for key in opened}
        if "main" in opened:
            side_note = "" if "side" in opened else " (측면 없음 — 메인만 녹화됩니다)"
            self.statusBar().showMessage(f"카메라 준비 완료{side_note}")
            self._record_button.setEnabled(True)
        else:
            self.statusBar().showMessage("메인 카메라를 열지 못했습니다. 장치 번호를 확인하세요")
            self._record_button.setEnabled(False)

    def _release_cameras(self) -> None:
        for key, cam in self._cams.items():
            if cam is not None:
                cam.release()
            self._cams[key] = None
        self._open_indices = set()

    def _scan_cameras(self) -> None:
        """인덱스 0~7을 프로브해 열리는 카메라와 해상도를 상태바에 표시한다.

        이미 열려 있는 카메라(메인/측면 프리뷰 중)는 프로브를 건너뛰어 방해하지
        않고 "(사용 중)"으로만 표시한다.
        """
        if self._recording:
            return
        self.statusBar().showMessage("카메라 검색 중... (수 초 걸릴 수 있음)")
        QApplication.processEvents()

        found = []
        for index in range(8):
            if index in self._open_indices:
                found.append(f"{index} (사용 중)")
                continue
            cam = _open_capture(index)
            if cam.isOpened():
                ok, frame = cam.read()
                if ok:
                    height, width = frame.shape[:2]
                    found.append(f"{index} ({width}x{height})")
            cam.release()

        if found:
            self.statusBar().showMessage(f"사용 가능한 카메라: {', '.join(found)}")
        else:
            self.statusBar().showMessage("사용 가능한 카메라를 찾지 못했습니다")

    # --- 검출 테스트 (녹화 전 각도 점검) ---

    def _on_detect_toggled(self, on: bool) -> None:
        """메인 프리뷰에 MediaPipe 손 검출 결과를 겹쳐 보여주는 점검 모드.

        추출기(tools/extract_landmarks.py)와 같은 신뢰도(0.7/0.6)를 쓰므로, 여기서
        손이 잡히면 녹화 영상에서도 잡힌다고 기대할 수 있다. 녹화 중에는 쓰지 않는다.
        """
        if not on:
            if self._test_tracker is not None:
                self._test_tracker.close()
                self._test_tracker = None
            return
        try:
            from core.tracker import HandTracker
        except ImportError:
            self._detect_button.setChecked(False)
            self.statusBar().showMessage(
                "검출 테스트를 쓰려면 mediapipe 설치와 core 리팩토링(PR #9) 이후 git pull이 필요합니다"
            )
            return
        self._test_tracker = HandTracker(
            max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6
        )
        self.statusBar().showMessage("검출 테스트 중 — 메인 프리뷰에서 손 검출 여부를 확인하세요")

    def _draw_detection(self, frame) -> None:
        """메인 프리뷰용 프레임에 손 검출 결과를 그린다 (녹화 파일에는 영향 없음)."""
        hands = self._test_tracker.process(frame)
        if hands:
            for px, py in hands[0].pixel_landmarks:
                cv2.circle(frame, (int(px), int(py)), 3, (0, 255, 0), -1)
            cv2.putText(frame, "HAND OK", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "NO HAND", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

    # --- 녹화 ---

    def _toggle_recording(self) -> None:
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        name = self._name_edit.text().strip()
        if not re.fullmatch(r"[A-Za-z0-9가-힣]+", name or ""):
            self.statusBar().showMessage("촬영자 이름을 입력하세요 (영문/한글/숫자만)")
            return

        # 검출 테스트는 녹화 FPS를 갉아먹으므로 녹화 시작 시 강제 종료한다.
        self._detect_button.setChecked(False)
        self._detect_button.setEnabled(False)

        light = self._light_box.currentText()
        speed = self._speed_box.currentText()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        take = next_take(OUTPUT_DIR, name, light, speed)
        base = build_basename(name, light, speed, take)

        for key, cam in self._cams.items():
            if cam is None:
                continue
            width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
            path = OUTPUT_DIR / f"{base}_{key}.mp4"
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), _NOMINAL_FPS, (width, height)
            )
            if not writer.isOpened():
                self.statusBar().showMessage(f"{path.name} 파일을 만들 수 없습니다")
                self._stop_recording()
                return
            self._writers[key] = writer

        self._csv_file = open(OUTPUT_DIR / f"{base}_frames.csv", "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(["frame_index", "elapsed_sec", "main_ok", "side_ok"])

        self._frame_index = 0
        self._record_start = time.perf_counter()
        self._recording = True
        self._base = base
        self._record_button.setText("■ 녹화 정지")
        self._set_form_enabled(False)

    def _stop_recording(self) -> None:
        self._recording = False
        for writer in self._writers.values():
            writer.release()
        self._writers.clear()
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
        self._record_button.setText("● 녹화 시작")
        self._set_form_enabled(True)
        self._detect_button.setEnabled(True)
        self.statusBar().showMessage(f"저장 완료: {OUTPUT_DIR}/{getattr(self, '_base', '')}_*")

    def _set_form_enabled(self, enabled: bool) -> None:
        for widget in (self._name_edit, self._light_box, self._speed_box, self._main_index, self._side_index):
            widget.setEnabled(enabled)

    # --- 프레임 루프 ---

    def _update_frames(self) -> None:
        frames = {}
        for key, cam in self._cams.items():
            if cam is None:
                continue
            ok, frame = cam.read()
            if ok:
                frames[key] = frame

        if self._recording:
            elapsed = time.perf_counter() - self._record_start
            for key, frame in frames.items():
                if key in self._writers:
                    self._writers[key].write(frame)
            if self._csv_writer is not None:
                main_ok = 1 if "main" in frames else 0
                side_ok = 1 if "side" in frames else 0
                self._csv_writer.writerow(
                    [self._frame_index, f"{elapsed:.4f}", main_ok, side_ok]
                )
            self._frame_index += 1
            self.statusBar().showMessage(f"녹화 중... {elapsed:.0f}초 ({self._frame_index} 프레임)")

        for key, frame in frames.items():
            view = self._previews[key]
            if key == "main" and self._test_tracker is not None and not self._recording:
                self._draw_detection(frame)
            if self._recording:
                cv2.circle(frame, (25, 25), 10, (0, 0, 255), -1)  # REC 표시
            view.setPixmap(to_pixmap(frame, view))

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 (Qt naming)
        self._timer.stop()
        if self._recording:
            self._stop_recording()
        if self._test_tracker is not None:
            self._test_tracker.close()
            self._test_tracker = None
        self._release_cameras()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    window = RecorderWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
