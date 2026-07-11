"""Main control loop: webcam -> hand tracking -> stroke rendering.

** week-2**에서는 keyboard 입력으로 pen up/down을 토글하도록 구현
- keyboard (SPACE)로 pen up/down을 토글합니다.
- keyboard (C)로 캔버스를 지웁니다.
- keyboard (S)로 캔버스를 저장합니다.
- keyboard (Q)로 프로그램을 종료합니다.

** week-3 에는 위에 key입력을 받지 않고, pen up/down을 손가락의 위치로 판단하여 stroke를 그리도록 구현할 예정

"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from core.tracker import HandTracker
from ui.canvas import StrokeCanvas

_STATUS_COLORS = {
    "PEN DOWN": (0, 200, 0),
    "PEN UP": (200, 200, 200),
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
    ) -> None:
        self._tracker = HandTracker(max_num_hands=1)
        self._line_color = line_color
        self._line_thickness = line_thickness
        self._canvas: StrokeCanvas | None = None
        self._pen_requested = False
        self._status = "PEN UP"

    @property
    def pen_requested(self) -> bool:
        return self._pen_requested

    @property
    def status(self) -> str:
        return self._status

    @property
    def canvas(self) -> StrokeCanvas | None:
        return self._canvas

    def set_pen_down(self, down: bool) -> None:
        """Request pen state; applied on the next frame with a tracked hand."""
        self._pen_requested = down

    def toggle_pen(self) -> bool:
        self._pen_requested = not self._pen_requested
        return self._pen_requested

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

        hands = self._tracker.process(bgr_frame)
        fingertip: tuple[int, int] | None = None
        if not hands:
            # Tracking loss must break the stroke, otherwise the next
            # detection draws a straight line from the stale point.
            self._canvas.pen_up()
            self._status = "HAND LOST"
        else:
            fingertip = hands[0].index_fingertip
            if self._pen_requested:
                if self._canvas.is_pen_down:
                    self._canvas.move(fingertip)
                else:
                    self._canvas.pen_down(fingertip)
                self._status = "PEN DOWN"
            else:
                self._canvas.pen_up()
                self._status = "PEN UP"

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
        return annotated

    def close(self) -> None:
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

    print("SPACE: 펜 토글 | C: 지우기 | S: 캔버스 저장 | Q: 종료")
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
                cv2.imshow("Virtual Whiteboard - SPACE pen / C clear / S save / Q quit", annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
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
