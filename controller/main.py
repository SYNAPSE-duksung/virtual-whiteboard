"""Main control loop: webcam -> hand tracking -> stroke rendering.

펜 판정은 두 모드를 ``M`` 키로 전환한다 (``WhiteboardSession.auto_mode``):
- 자동(AUTO): ``core.PenTracker``가 손끝 위치(pen_ratio)로 pen up/down을 판정하고,
  손을 펴면(open hand) 캔버스를 지운다.
- 수동(MANUAL): 기존 방식. SPACE로 pen up/down을 토글하고 C로 지운다 (자동 판정 무시).

키:
- keyboard (M): 자동/수동 모드 전환
- keyboard (SPACE): (수동 모드) pen up/down 토글
- keyboard (R): 좌표 CSV 기록 on/off (output/coords_*.csv)
- keyboard (C): 캔버스를 지웁니다.
- keyboard (S): 캔버스를 저장합니다.
- keyboard (Q): 프로그램을 종료합니다.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from core.pen_tracker import PenTracker
from core.recorder import CoordRecorder, CoordSample
from ui.canvas import StrokeCanvas

_STATUS_COLORS = {
    "PEN DOWN": (0, 200, 0),
    "PEN UP": (200, 200, 200),
    "ERASE": (0, 215, 255),
    "HAND LOST": (0, 100, 255),
}


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
        output_dir: str | Path = "output",
    ) -> None:
        self._tracker = PenTracker(
            min_cutoff=min_cutoff,
            beta=beta,
            pen_down_thresh=pen_down_thresh,
            pen_up_thresh=pen_up_thresh,
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
        if self._canvas is None:
            self._canvas = StrokeCanvas(
                width,
                height,
                line_color=self._line_color,
                line_thickness=self._line_thickness,
            )

        result = self._tracker.process(bgr_frame)
        fingertip = result.fingertip
        if not result.hand_detected:
            # Tracking loss must break the stroke, otherwise the next
            # detection draws a straight line from the stale point.
            self._canvas.pen_up()
            self._status = "HAND LOST"
        elif self._auto_mode:
            # 자동 모드: pen_ratio 자동 판정 + 손 펴기 지우기 (PenTracker).
            if result.erase_gesture:
                self._canvas.clear()
                self._status = "ERASE"
            elif result.pen_down:
                self._draw_to(fingertip)
                self._status = "PEN DOWN"
            else:
                self._canvas.pen_up()
                self._status = "PEN UP"
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
        self._recorder.close()
        self._tracker.close()

    def __enter__(self) -> "WhiteboardSession":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="camera device index")
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="flip the frame horizontally (selfie view; off by default for the desk view)",
    )
    args = parser.parse_args()

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        print(f"카메라 {args.camera}을(를) 열 수 없습니다.")
        return 1

    print("M: 모드 전환 | (자동)손끝 판정+손 펴기 지우기 / (수동)SPACE 펜 | R: CSV 기록 | C: 지우기 | S: 저장 | Q: 종료")
    previous_time = time.perf_counter()
    try:
        with WhiteboardSession() as session:
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("카메라 프레임을 읽지 못했습니다.")
                    return 1
                if args.mirror:
                    frame = cv2.flip(frame, 1)

                annotated = session.process_frame(frame)

                current_time = time.perf_counter()
                fps = 1.0 / max(current_time - previous_time, 1e-6)
                previous_time = current_time
                cv2.putText(annotated, f"FPS {fps:.1f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("Virtual Whiteboard - M mode / SPACE pen / R rec / C clear / S save / Q quit", annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("m"):
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
