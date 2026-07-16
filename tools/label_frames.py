"""측면(정답 판정용) 녹화 영상을 보며 pen-up/down 정답 라벨을 만드는 도구 (D파트 학습 라벨용).

``tools/record_dual.py``로 촬영한 ``{base}_side.mp4``를 재생하면서, 손끝이 책상면에
닿는/떨어지는 순간마다 **토글 지점**을 찍으면 전체 프레임의 pen_down(0/1) 라벨이 자동으로
전개된다. 같은 폴더의 ``{base}_main.mp4``(문맥 확인용)와 ``{base}_frames.csv``(두 영상의
프레임 정렬 기준)가 있으면 함께 활용한다.

라벨링 모델
-----------
필기는 down/up 전환이 잦으므로 프레임 하나하나를 클릭하는 대신, 상태가 바뀌는 순간에만
``T`` 키(또는 버튼)로 전환점을 찍는다. 초기 상태는 항상 UP이고, 전환점들이 오름차순으로
상태를 번갈아 뒤집어 전체 프레임 라벨이 결정된다(토글 프레임 자체부터 새 상태 적용).

프레임 정렬
-----------
``tools/record_dual.py``는 틱마다 두 카메라를 읽되 못 읽은 쪽은 영상에 쓰지 않으므로,
영상 파일 내부의 프레임 번호와 녹화 시점의 전역 ``frame_index``가 어긋날 수 있다.
``{base}_frames.csv``(헤더 ``frame_index,elapsed_sec,main_ok,side_ok``)에서
``side_ok == 1`` 행의 frame_index 목록이 "측면 영상의 k번째 프레임 -> 전역 frame_index"
매핑이고, ``main_ok == 1`` 행의 frame_index 목록으로 전역 인덱스에 가장 가까운(그 이전)
메인 프레임을 찾는다. 토글 지점과 출력 라벨은 이 전역 frame_index 기준이다.
frames.csv가 없으면 측면 영상의 프레임 번호를 그대로 전역 frame_index로 간주한다
(상태바에 경고 표시).

카메라 지연 오프셋 보정
------------------------
측면 카메라(폰, Iriun 경유)는 파이프라인 지연 때문에 같은 frame_index에 기록된 메인
프레임보다 과거 장면을 담는다. 이를 보정하기 위해 토글 지점은 항상 **측면 영상 프레임
인덱스 k 공간**으로 내부 보관한다(라벨러가 "본 순간"에 앵커되어, 오프셋을 나중에 바꿔도
이미 찍은 토글이 오염되지 않는다). "지연 오프셋(프레임)" 스핀박스로 지연 프레임 수를
지정하면, 측면 프레임 k는 ``g = side_map[k] - offset``으로 보정된 전역 frame_index에
대응한다고 간주해 메인 미니뷰와 저장 라벨을 계산한다. 저장 시에만 k 공간 토글을
``side_toggles_to_global``로 전역 공간으로 변환해 ``expand_labels``에 넘긴다. 오프셋 값은
영상과 같은 폴더의 ``labeling_offsets.json``에 ``{base: offset}`` 형태로 저장되어, 다음에
같은 영상을 열 때 자동 복원된다.

출력
-----
``{base}_labels.csv`` (영상과 같은 폴더, 헤더 ``video_id,frame_id,pen_down``) — 전역
프레임마다 한 행. D파트가 ``{base}_coords.csv``와 ``video_id`` + ``frame_id``로 조인한다.
기존 라벨 파일이 있으면 로드해 토글 지점을 복원한 뒤 이어서 편집할 수 있다.
``labeling_offsets.json`` (같은 폴더, 여러 영상이 공유하는 사이드카) — ``{base: offset}``
매핑. 저장할 때마다 현재 base의 오프셋 값이 갱신되고, 다른 base 항목은 보존된다.

mediapipe는 필요하지 않다 (이미 녹화된 영상 + CSV만 다룬다).

실행: python -m tools.label_frames
조작: SPACE 재생/정지 · ←/→ 1프레임(Shift+←/→ 10프레임) · T 토글 마크 ·
      Ctrl+Z 마지막 토글 취소 · Ctrl+S 저장 ·
      "지연 오프셋(프레임)" 스핀박스(-30~30) — 박수 등 동기화 순간을 찾은 뒤 좌측 메인
      화면이 같은 순간을 보일 때까지 값을 조절해 측면 카메라 지연을 보정한다
"""

from __future__ import annotations

import bisect
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QCloseEvent, QColor, QImage, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

OUTPUT_DIR = Path("data/dataset/recordings")
SIDE_SUFFIX = "_side.mp4"
MAIN_SUFFIX = "_main.mp4"
FRAMES_SUFFIX = "_frames.csv"
LABELS_SUFFIX = "_labels.csv"
OFFSETS_FILENAME = "labeling_offsets.json"

_DEFAULT_FPS = 30.0
_SPEED_OPTIONS = ("0.5x", "1x", "2x")
_OFFSET_MIN = -30
_OFFSET_MAX = 30


# --------------------------------------------------------------------------
# 순수 로직 (GUI 없이 테스트 가능)
# --------------------------------------------------------------------------


@dataclass
class FrameMaps:
    """frames.csv로부터 만든 전역 <-> 측면/메인 영상 프레임 매핑표."""

    side_map: list[int]  # side_map[k] == 측면 영상 k번째 프레임의 전역 frame_index
    main_map: list[int]  # main_map[i] == 메인 영상 i번째 프레임의 전역 frame_index
    total_frames: int  # 전역 frame_index 총 개수 (frames.csv 행 수 기준)
    has_frames_csv: bool  # False면 항등 매핑으로 대체된 상태


def video_base_from_side(path: Path) -> str:
    """``{base}_side.mp4``에서 base명을 뽑는다 (video_id로도 재사용)."""
    name = path.name
    if name.endswith(SIDE_SUFFIX):
        return name[: -len(SIDE_SUFFIX)]
    return path.stem


def load_frame_maps(frames_csv: Path | None, side_frame_count: int) -> FrameMaps:
    """frames.csv를 읽어 전역 frame_index <-> 측면/메인 프레임 매핑표를 만든다.

    frames_csv가 None이거나 실제로 존재하지 않으면, 측면 영상의 프레임 번호를 그대로
    전역 frame_index로 쓰는 항등 매핑으로 대체한다(메인 매핑은 알 수 없으므로 빈 목록).
    구형(``frame_index,elapsed_sec`` 2컬럼) frames.csv는 모든 프레임을
    main_ok=side_ok=1로 간주해 지원한다.
    """
    if frames_csv is None or not frames_csv.exists():
        return FrameMaps(
            side_map=list(range(side_frame_count)),
            main_map=[],
            total_frames=side_frame_count,
            has_frames_csv=False,
        )

    side_map: list[int] = []
    main_map: list[int] = []
    max_index = -1
    with open(frames_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        legacy = "side_ok" not in fieldnames or "main_ok" not in fieldnames
        for row in reader:
            try:
                idx = int(row["frame_index"])
            except (KeyError, ValueError, TypeError):
                continue
            max_index = max(max_index, idx)
            if legacy:
                side_ok = main_ok = 1
            else:
                side_ok = int(row.get("side_ok") or 0)
                main_ok = int(row.get("main_ok") or 0)
            if side_ok:
                side_map.append(idx)
            if main_ok:
                main_map.append(idx)

    side_map.sort()
    main_map.sort()
    return FrameMaps(
        side_map=side_map,
        main_map=main_map,
        total_frames=max_index + 1 if max_index >= 0 else 0,
        has_frames_csv=True,
    )


def global_to_side_index(side_map: list[int], global_frame: int) -> int:
    """전역 frame_index에 가장 가까운 측면 영상 프레임 번호(0-based)를 구한다."""
    if not side_map:
        return 0
    pos = bisect.bisect_left(side_map, global_frame)
    if pos >= len(side_map):
        pos = len(side_map) - 1
    return pos


def global_to_main_index(main_map: list[int], global_frame: int) -> int | None:
    """전역 frame_index에 대응하는 메인 영상 프레임 번호를 구한다.

    main_map에서 global_frame 이하인 값 중 마지막 것의 위치를 돌려준다(그 시점 메인
    프레임이 누락됐으면 직전 프레임을 그대로 보여준다는 규칙). 대응하는 메인 프레임이
    아직 없으면(가장 이른 메인 프레임보다도 앞이면) None.
    """
    if not main_map:
        return None
    pos = bisect.bisect_right(main_map, global_frame) - 1
    return pos if pos >= 0 else None


def expand_labels(toggles: list[int], total: int) -> list[int]:
    """토글 지점 목록을 전체 프레임 pen_down(0/1) 배열로 전개한다.

    초기 상태는 UP(0). 토글 프레임 자체부터 반전된 상태가 적용된다.
    """
    ts = sorted({g for g in toggles if 0 <= g < total})
    labels = [0] * total
    state = 0
    prev_boundary = 0
    for t in ts:
        if state == 1:
            for i in range(prev_boundary, t):
                labels[i] = 1
        state ^= 1
        prev_boundary = t
    if state == 1:
        for i in range(prev_boundary, total):
            labels[i] = 1
    return labels


def toggles_from_labels(labels: list[int]) -> list[int]:
    """0/1 라벨 배열에서 전환점(토글 지점) 목록을 복원한다.

    프레임 0 이전의 가상 상태는 UP(0)으로 간주한다. ``expand_labels``의 역연산.
    """
    toggles: list[int] = []
    prev = 0
    for i, value in enumerate(labels):
        if value != prev:
            toggles.append(i)
            prev = value
    return toggles


def down_regions(toggles: list[int], total: int) -> list[tuple[int, int]]:
    """토글 지점들로부터 pen-down 구간 목록 ``[(시작, 끝) , ...]``(끝은 미포함)을 만든다."""
    ts = sorted({g for g in toggles if 0 <= g < total})
    regions: list[tuple[int, int]] = []
    for i in range(0, len(ts), 2):
        start = ts[i]
        end = ts[i + 1] if i + 1 < len(ts) else total
        regions.append((start, end))
    return regions


def format_toggle_label(order_index: int, k: int, g: int) -> str:
    """토글 지점 목록 UI 항목 문자열(``측면 프레임 k (전역 g): UP→DOWN``)을 만든다.

    k는 측면 영상 프레임 번호(내부 보관 공간), g는 지연 오프셋을 적용해 보정한
    전역 frame_index(저장 시 실제로 쓰이는 값)다.
    """
    before = "UP" if order_index % 2 == 0 else "DOWN"
    after = "DOWN" if order_index % 2 == 0 else "UP"
    return f"측면 프레임 {k} (전역 {g}): {before}→{after}"


def write_labels_csv(path: Path, video_id: str, labels: list[int]) -> None:
    """``video_id,frame_id,pen_down`` 스키마로 전역 프레임당 한 행씩 저장한다."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "frame_id", "pen_down"])
        for frame_id, pen_down in enumerate(labels):
            writer.writerow([video_id, frame_id, int(pen_down)])


def read_labels_csv(path: Path) -> tuple[str, list[int]]:
    """기존 라벨 CSV를 읽어 ``(video_id, labels)``를 돌려준다.

    frame_id가 누락된 경우를 대비해 최댓값+1 길이의 배열에 채워 넣는다(누락 행은 0).
    """
    rows: list[tuple[int, int]] = []
    video_id = ""
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame_id = int(row["frame_id"])
                pen_down = int(row["pen_down"])
            except (KeyError, ValueError, TypeError):
                continue
            video_id = row.get("video_id") or video_id
            rows.append((frame_id, pen_down))
    if not rows:
        return video_id, []
    max_frame = max(frame_id for frame_id, _ in rows)
    labels = [0] * (max_frame + 1)
    for frame_id, pen_down in rows:
        labels[frame_id] = pen_down
    return video_id, labels


def side_toggles_to_global(
    toggles_k: list[int], side_map: list[int], offset: int, total: int
) -> list[int]:
    """측면 공간 토글 k 목록을 저장용 전역 frame_index 목록으로 변환한다.

    각 k에 대해 ``g = side_map[k] - offset``을 계산해 ``[0, total - 1]`` 범위로
    클램프한다. side_map 범위 밖의 k는 무시한다. 클램프로 인해 서로 다른 k가 같은 g로
    겹치면 정렬 후 중복 제거해 병합한다.
    """
    if total <= 0:
        return []
    n = len(side_map)
    result: set[int] = set()
    for k in toggles_k:
        if k < 0 or k >= n:
            continue
        g = side_map[k] - offset
        g = max(0, min(g, total - 1))
        result.add(g)
    return sorted(result)


def global_toggles_to_side(toggles_g: list[int], side_map: list[int], offset: int) -> list[int]:
    """전역 토글 목록을 측면 공간 k 목록으로 역변환한다 (기존 라벨 이어서 편집할 때 사용).

    각 g에 대해 ``side_map[k] ≈ g + offset``을 만족하는 k를 이분탐색으로 가장 가까운
    값에서 찾는다. ``side_toggles_to_global``의 근사 역연산이며, 클램프가 걸리지 않은
    범위에서는 정확히 원래 k로 복원된다.
    """
    if not side_map:
        return []
    n = len(side_map)
    result: set[int] = set()
    for g in toggles_g:
        target = g + offset
        pos = bisect.bisect_left(side_map, target)
        candidates = [idx for idx in (pos - 1, pos) if 0 <= idx < n]
        if not candidates:
            continue
        best = min(candidates, key=lambda idx: abs(side_map[idx] - target))
        result.add(best)
    return sorted(result)


def load_labeling_offsets(path: Path) -> dict[str, int]:
    """``labeling_offsets.json`` 사이드카를 읽어 ``{base: offset}`` 딕셔너리를 돌려준다.

    파일이 없거나 JSON 파싱에 실패하면(손상된 파일 포함) 빈 딕셔너리로 폴백한다.
    """
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    result: dict[str, int] = {}
    for key, value in data.items():
        try:
            result[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return result


def save_labeling_offset(path: Path, base: str, offset: int) -> None:
    """``labeling_offsets.json``에 ``base``의 오프셋 값을 기록한다.

    기존 파일의 다른 base 항목은 보존한 채 병합 저장한다.
    """
    data = load_labeling_offsets(path)
    data[base] = int(offset)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------


def _to_pixmap(bgr_frame, target: QLabel) -> QPixmap:
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    height, width, _ = rgb.shape
    image = QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image).scaled(
        target.size(),
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


class TimelineWidget(QWidget):
    """pen-down 구간을 빨간 띠로, 현재 위치를 세로선으로 그리는 하단 타임라인.

    모든 좌표는 측면 영상 프레임 인덱스(k) 공간 기준이다(토글 목록도, 현재 위치도).
    전역 frame_index로의 변환은 저장 시에만 일어나므로 이 위젯은 신경 쓰지 않는다.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(28)
        self._toggles_k: list[int] = []
        self._total_k: int = 0
        self._current_k: int = 0
        self.on_seek: Callable[[int], None] | None = None

    def set_data(self, toggles_k: list[int], total_k: int, current_k: int) -> None:
        self._toggles_k = toggles_k
        self._total_k = total_k
        self._current_k = current_k
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        width, height = self.width(), self.height()
        painter.fillRect(0, 0, width, height, QColor(60, 60, 60))
        n = self._total_k
        if n > 0:
            for start_k, end_k in down_regions(self._toggles_k, n):
                px0 = int(start_k / n * width)
                px1 = int(end_k / n * width)
                painter.fillRect(px0, 0, max(px1 - px0, 1), height, QColor(210, 40, 40))
            cx = int(self._current_k / (n - 1) * width) if n > 1 else 0
            painter.setPen(QColor(255, 220, 0))
            painter.drawLine(cx, 0, cx, height)
        painter.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        n = self._total_k
        if n <= 1 or self.on_seek is None:
            return
        ratio = max(0.0, min(1.0, event.position().x() / max(self.width(), 1)))
        self.on_seek(int(round(ratio * (n - 1))))


class LabelWindow(QMainWindow):
    """측면 영상 프리뷰 + 메인 프리뷰 + 토글 편집 패널 + 타임라인을 담은 라벨링 도구."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("데이터 라벨링 - Pen Up/Down")

        self._side_cap: cv2.VideoCapture | None = None
        self._main_cap: cv2.VideoCapture | None = None
        self._side_map: list[int] = []
        self._main_map: list[int] = []
        self._total_frames = 0
        self._has_frames_csv = True
        self._toggles: list[int] = []
        self._undo_stack: list[int] = []
        self._current_k = 0
        self._last_side_frame = None
        self._base = ""
        self._folder = OUTPUT_DIR
        self._labels_path: Path | None = None
        self._dirty = False
        self._playing = False
        self._fps = _DEFAULT_FPS
        self._speed = 1.0
        self._updating_slider = False

        # --- 상단: 영상 선택 ---
        self._video_combo = QComboBox()
        self._video_combo.currentIndexChanged.connect(self._on_combo_changed)
        open_button = QPushButton("파일 열기")
        open_button.clicked.connect(self._on_open_file)
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("측면 영상"))
        top_bar.addWidget(self._video_combo, stretch=1)
        top_bar.addWidget(open_button)

        # --- 좌측: 메인 프리뷰 + 편집 패널 ---
        self._main_view = QLabel("메인 영상 없음")
        self._main_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._main_view.setMinimumSize(320, 240)

        self._toggle_list = QListWidget()
        delete_button = QPushButton("삭제")
        delete_button.clicked.connect(self._delete_selected)
        minus_button = QPushButton("◀ -1프레임")
        minus_button.clicked.connect(lambda: self._shift_selected(-1))
        plus_button = QPushButton("+1프레임 ▶")
        plus_button.clicked.connect(lambda: self._shift_selected(1))
        edit_buttons = QHBoxLayout()
        edit_buttons.addWidget(delete_button)
        edit_buttons.addWidget(minus_button)
        edit_buttons.addWidget(plus_button)

        self._offset_spin = QSpinBox()
        self._offset_spin.setRange(_OFFSET_MIN, _OFFSET_MAX)
        self._offset_spin.setValue(0)
        self._offset_spin.setToolTip(
            "측면 영상이 메인보다 늦게 도착한 프레임 수. 박수 순간을 슬라이더로 찾은 뒤, "
            "좌측 메인 화면이 같은 순간을 보일 때까지 이 값을 조절하세요."
        )
        self._offset_spin.valueChanged.connect(self._on_offset_changed)
        offset_row = QHBoxLayout()
        offset_row.addWidget(QLabel("지연 오프셋(프레임)"))
        offset_row.addWidget(self._offset_spin)
        offset_row.addStretch(1)

        self._label_file_display = QLabel("라벨 파일: -")

        left_col = QVBoxLayout()
        left_col.addWidget(QLabel("메인 (사선) - 문맥 확인용"))
        left_col.addWidget(self._main_view)
        left_col.addLayout(offset_row)
        left_col.addWidget(QLabel("토글 지점"))
        left_col.addWidget(self._toggle_list, stretch=1)
        left_col.addLayout(edit_buttons)
        left_col.addWidget(self._label_file_display)

        # --- 우측: 측면(정답 판정 기준) 프리뷰 ---
        self._side_view = QLabel("측면 영상을 선택하세요")
        self._side_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._side_view.setMinimumSize(640, 480)
        right_col = QVBoxLayout()
        right_col.addWidget(QLabel("측면 - 라벨 판정 기준 화면"))
        right_col.addWidget(self._side_view, stretch=1)

        content = QHBoxLayout()
        content.addLayout(left_col, stretch=1)
        content.addLayout(right_col, stretch=2)

        # --- 하단: 타임라인 + 슬라이더 + 재생 제어 ---
        self._timeline = TimelineWidget()
        self._timeline.on_seek = self._seek

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.valueChanged.connect(self._on_slider_changed)
        self._frame_label = QLabel("0 / 0")
        slider_row = QHBoxLayout()
        slider_row.addWidget(self._slider, stretch=1)
        slider_row.addWidget(self._frame_label)

        self._play_button = QPushButton("재생 (Space)")
        self._play_button.clicked.connect(self._toggle_play)
        self._speed_combo = QComboBox()
        self._speed_combo.addItems(list(_SPEED_OPTIONS))
        self._speed_combo.setCurrentText("1x")
        self._speed_combo.currentTextChanged.connect(self._on_speed_changed)
        mark_button = QPushButton("토글 마크 (T)")
        mark_button.clicked.connect(self._mark_toggle)
        undo_button = QPushButton("토글 취소 (Ctrl+Z)")
        undo_button.clicked.connect(self._undo_last_toggle)
        save_button = QPushButton("저장 (Ctrl+S)")
        save_button.clicked.connect(self._save)
        controls_row = QHBoxLayout()
        controls_row.addWidget(self._play_button)
        controls_row.addWidget(self._speed_combo)
        controls_row.addWidget(mark_button)
        controls_row.addWidget(undo_button)
        controls_row.addStretch(1)
        controls_row.addWidget(save_button)

        bottom = QVBoxLayout()
        bottom.addWidget(self._timeline)
        bottom.addLayout(slider_row)
        bottom.addLayout(controls_row)

        layout = QVBoxLayout()
        layout.addLayout(top_bar)
        layout.addLayout(content, stretch=1)
        layout.addLayout(bottom)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.statusBar().showMessage("측면 영상을 선택하세요")

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer_tick)

        self._populate_video_list()

    # --- 영상 목록 / 로딩 ---

    def _populate_video_list(self) -> None:
        self._video_combo.blockSignals(True)
        self._video_combo.clear()
        self._video_combo.addItem("(측면 영상 선택)", None)
        if OUTPUT_DIR.exists():
            for path in sorted(OUTPUT_DIR.glob(f"*{SIDE_SUFFIX}")):
                self._video_combo.addItem(path.name, str(path))
        self._video_combo.blockSignals(False)

    def _on_combo_changed(self, _index: int) -> None:
        path_str = self._video_combo.currentData()
        if path_str:
            self._load_video(Path(path_str))

    def _on_open_file(self) -> None:
        start_dir = str(OUTPUT_DIR) if OUTPUT_DIR.exists() else "."
        path_str, _ = QFileDialog.getOpenFileName(
            self, "측면 영상 열기", start_dir, f"측면 영상 (*{SIDE_SUFFIX});;모든 파일 (*)"
        )
        if path_str:
            self._load_video(Path(path_str))

    def _confirm_discard(self, message: str) -> bool:
        if not self._dirty:
            return True
        choice = QMessageBox.question(
            self,
            "확인",
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        return choice == QMessageBox.StandardButton.Yes

    def _load_video(self, path: Path) -> None:
        if not self._confirm_discard("저장하지 않은 변경사항이 있습니다. 다른 영상을 열까요?"):
            return

        self._set_playing(False)
        if self._side_cap is not None:
            self._side_cap.release()
        if self._main_cap is not None:
            self._main_cap.release()
            self._main_cap = None

        side_cap = cv2.VideoCapture(str(path))
        if not side_cap.isOpened():
            self.statusBar().showMessage(f"측면 영상을 열 수 없습니다: {path}")
            self._side_cap = None
            return
        self._side_cap = side_cap

        side_frame_count = int(side_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = side_cap.get(cv2.CAP_PROP_FPS)
        self._fps = fps if fps and fps > 0 else _DEFAULT_FPS

        base = video_base_from_side(path)
        folder = path.parent
        self._base = base
        self._folder = folder

        frames_csv = folder / f"{base}{FRAMES_SUFFIX}"
        maps = load_frame_maps(frames_csv if frames_csv.exists() else None, side_frame_count)
        self._side_map = maps.side_map
        self._main_map = maps.main_map
        self._total_frames = maps.total_frames
        self._has_frames_csv = maps.has_frames_csv

        main_path = folder / f"{base}{MAIN_SUFFIX}"
        self._main_view.setText("메인 영상 없음")
        if main_path.exists():
            main_cap = cv2.VideoCapture(str(main_path))
            if main_cap.isOpened():
                self._main_cap = main_cap
                if not maps.has_frames_csv:
                    # frames.csv 없이 메인 영상만 있는 경우: 항등 매핑으로 대체
                    main_count = int(main_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    self._main_map = list(range(main_count))
            else:
                main_cap.release()
                self._main_view.setText("메인 영상 없음 (열기 실패)")

        offsets_path = folder / OFFSETS_FILENAME
        offsets = load_labeling_offsets(offsets_path)
        offset_known = base in offsets
        offset = offsets.get(base, 0)
        self._offset_spin.blockSignals(True)
        self._offset_spin.setValue(offset)
        self._offset_spin.blockSignals(False)

        self._labels_path = folder / f"{base}{LABELS_SUFFIX}"
        offset_note = ""
        if self._labels_path.exists():
            _video_id, labels = read_labels_csv(self._labels_path)
            if labels and len(labels) != self._total_frames:
                self.statusBar().showMessage(
                    f"경고: 저장된 라벨 프레임 수({len(labels)})가 현재 총 프레임 수"
                    f"({self._total_frames})와 다릅니다. 더 큰 쪽을 기준으로 이어서 편집합니다"
                )
                self._total_frames = max(self._total_frames, len(labels))
            toggles_g = toggles_from_labels(labels)
            self._toggles = global_toggles_to_side(toggles_g, self._side_map, offset)
            if not offset_known and toggles_g:
                offset_note = " (labeling_offsets.json에 오프셋 기록 없음 — 0으로 간주)"
        else:
            self._toggles = []
        self._undo_stack = []
        self._dirty = False
        self._current_k = 0

        self._slider.setMaximum(max(len(self._side_map) - 1, 0))
        self._label_file_display.setText(f"라벨 파일: {self._labels_path.name}")
        self._refresh_after_edit()
        if self._side_map:
            self._seek(0)

        note = "" if maps.has_frames_csv else " (frames.csv 없음 — 측면 프레임 번호를 그대로 전역 인덱스로 사용)"
        self.statusBar().showMessage(f"{path.name} 로드 완료{note}{offset_note}")

    # --- 프레임 탐색 / 표시 ---

    def _corrected_global_for_k(self, k: int) -> int:
        """측면 프레임 k를 지연 오프셋으로 보정한 전역 frame_index로 변환한다.

        ``g = side_map[k] - offset``를 ``[0, total_frames - 1]`` 범위로 클램프한다.
        """
        if not self._side_map:
            return k
        k = max(0, min(k, len(self._side_map) - 1))
        g = self._side_map[k] - self._offset_spin.value()
        return max(0, min(g, max(self._total_frames - 1, 0)))

    def _current_corrected_global(self) -> int:
        return self._corrected_global_for_k(self._current_k)

    def _current_pen_down(self) -> bool:
        return bool(bisect.bisect_right(sorted(self._toggles), self._current_k) % 2)

    def _seek(self, k: int) -> None:
        if self._side_cap is None or not self._side_map:
            return
        k = max(0, min(k, len(self._side_map) - 1))
        self._current_k = k
        self._side_cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, frame = self._side_cap.read()
        if not ok:
            return
        self._last_side_frame = frame
        self._render_side_frame()
        self._render_main_frame()
        self._update_slider_position()
        self._update_frame_label()
        self._timeline.set_data(self._toggles, len(self._side_map), self._current_k)

    def _render_side_frame(self) -> None:
        if self._last_side_frame is None:
            return
        frame = self._last_side_frame.copy()
        down = self._current_pen_down()
        text = "PEN DOWN" if down else "PEN UP"
        color = (0, 0, 255) if down else (160, 160, 160)
        cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 4, cv2.LINE_AA)
        self._side_view.setPixmap(_to_pixmap(frame, self._side_view))

    def _render_main_frame(self) -> None:
        if self._main_cap is None:
            return
        idx = global_to_main_index(self._main_map, self._current_corrected_global())
        if idx is None:
            return
        self._main_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self._main_cap.read()
        if ok:
            self._main_view.setPixmap(_to_pixmap(frame, self._main_view))

    def _update_slider_position(self) -> None:
        self._updating_slider = True
        self._slider.setValue(self._current_k)
        self._updating_slider = False

    def _update_frame_label(self) -> None:
        g = self._current_corrected_global()
        self._frame_label.setText(f"{g} / {max(self._total_frames - 1, 0)}")

    def _on_slider_changed(self, value: int) -> None:
        if self._updating_slider:
            return
        self._seek(value)

    # --- 재생 ---

    def _toggle_play(self) -> None:
        self._set_playing(not self._playing)

    def _set_playing(self, playing: bool) -> None:
        if playing and not self._side_map:
            return
        self._playing = playing
        if playing:
            interval = max(1, int(1000 / (self._fps * self._speed)))
            self._timer.start(interval)
            self._play_button.setText("일시정지 (Space)")
        else:
            self._timer.stop()
            self._play_button.setText("재생 (Space)")

    def _on_timer_tick(self) -> None:
        if self._current_k + 1 >= len(self._side_map):
            self._set_playing(False)
            return
        self._seek(self._current_k + 1)

    def _on_speed_changed(self, text: str) -> None:
        try:
            self._speed = float(text.rstrip("x"))
        except ValueError:
            self._speed = 1.0
        if self._playing:
            self._set_playing(True)

    # --- 지연 오프셋 ---

    def _on_offset_changed(self, _value: int) -> None:
        self._dirty = True
        self._render_main_frame()
        self._refresh_after_edit()

    # --- 토글 편집 (모두 측면 프레임 인덱스 k 공간에서 동작) ---

    def _mark_toggle(self) -> None:
        if not self._side_map:
            return
        k = self._current_k
        if k in self._toggles:
            self._toggles.remove(k)
            if k in self._undo_stack:
                self._undo_stack.remove(k)
        else:
            bisect.insort(self._toggles, k)
            self._undo_stack.append(k)
        self._dirty = True
        self._refresh_after_edit()

    def _undo_last_toggle(self) -> None:
        if not self._undo_stack:
            return
        k = self._undo_stack.pop()
        if k in self._toggles:
            self._toggles.remove(k)
        self._dirty = True
        self._refresh_after_edit()

    def _delete_selected(self) -> None:
        row = self._toggle_list.currentRow()
        if row < 0 or row >= len(self._toggles):
            return
        k = self._toggles.pop(row)
        if k in self._undo_stack:
            self._undo_stack.remove(k)
        self._dirty = True
        self._refresh_after_edit()

    def _shift_selected(self, delta: int) -> None:
        row = self._toggle_list.currentRow()
        if row < 0 or row >= len(self._toggles):
            return
        k = self._toggles[row]
        new_k = max(0, min(k + delta, len(self._side_map) - 1)) if self._side_map else k
        if new_k == k:
            return
        if new_k in self._toggles:
            self.statusBar().showMessage("이미 같은 프레임에 토글이 있습니다")
            return
        self._toggles[row] = new_k
        self._toggles.sort()
        if k in self._undo_stack:
            self._undo_stack[self._undo_stack.index(k)] = new_k
        self._dirty = True
        self._refresh_after_edit()
        self._seek(new_k)

    def _refresh_after_edit(self) -> None:
        self._toggle_list.clear()
        for i, k in enumerate(self._toggles):
            g = self._corrected_global_for_k(k)
            self._toggle_list.addItem(format_toggle_label(i, k, g))
        self._timeline.set_data(self._toggles, len(self._side_map), self._current_k)
        self._render_side_frame()

    # --- 저장 ---

    def _save(self) -> None:
        if self._labels_path is None:
            return
        offset = self._offset_spin.value()
        toggles_g = side_toggles_to_global(self._toggles, self._side_map, offset, self._total_frames)
        labels = expand_labels(toggles_g, self._total_frames)
        if labels and labels[-1] == 1:
            choice = QMessageBox.question(
                self,
                "저장 확인",
                "마지막 상태가 PEN DOWN입니다. 이대로 저장할까요?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if choice != QMessageBox.StandardButton.Yes:
                return
        write_labels_csv(self._labels_path, self._base, labels)
        save_labeling_offset(self._folder / OFFSETS_FILENAME, self._base, offset)
        self._dirty = False
        self.statusBar().showMessage(
            f"저장 완료: {self._labels_path.name} ({len(labels)}프레임, 오프셋 {offset}프레임 적용)"
        )

    # --- 입력 ---

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        key = event.key()
        mods = event.modifiers()
        if key == Qt.Key.Key_Space:
            self._toggle_play()
        elif key == Qt.Key.Key_Left:
            step = 10 if mods & Qt.KeyboardModifier.ShiftModifier else 1
            self._seek(self._current_k - step)
        elif key == Qt.Key.Key_Right:
            step = 10 if mods & Qt.KeyboardModifier.ShiftModifier else 1
            self._seek(self._current_k + step)
        elif key == Qt.Key.Key_T:
            self._mark_toggle()
        elif key == Qt.Key.Key_Z and (mods & Qt.KeyboardModifier.ControlModifier):
            self._undo_last_toggle()
        elif key == Qt.Key.Key_S and (mods & Qt.KeyboardModifier.ControlModifier):
            self._save()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 (Qt naming)
        if not self._confirm_discard("저장하지 않은 변경사항이 있습니다. 종료할까요?"):
            event.ignore()
            return
        self._timer.stop()
        if self._side_cap is not None:
            self._side_cap.release()
        if self._main_cap is not None:
            self._main_cap.release()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    window = LabelWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
