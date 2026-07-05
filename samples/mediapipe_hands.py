"""Webcam demo: draw MediaPipe hand landmarks and index fingertip position."""

from __future__ import annotations

import argparse
import time

import cv2

from core.tracker import HandTracker, TrackedHand


CONNECTIONS = tuple((int(a), int(b)) for a, b in __import__("mediapipe").solutions.hands.HAND_CONNECTIONS)


def draw_hand(frame, hand: TrackedHand) -> None:
    for start, end in CONNECTIONS:
        cv2.line(frame, tuple(hand.pixel_landmarks[start]), tuple(hand.pixel_landmarks[end]), (80, 220, 80), 2)
    for x, y in hand.pixel_landmarks:
        cv2.circle(frame, (int(x), int(y)), 3, (30, 80, 255), -1)

    tip_x, tip_y = hand.index_fingertip
    cv2.circle(frame, (tip_x, tip_y), 10, (255, 180, 0), 2)
    cv2.putText(
        frame,
        f"{hand.handedness} {hand.score:.2f} | index=({tip_x}, {tip_y})",
        (tip_x + 12, max(25, tip_y - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="camera device index")
    args = parser.parse_args()

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        print(f"카메라 {args.camera}을(를) 열 수 없습니다.")
        return 1

    previous_time = time.perf_counter()
    try:
        with HandTracker(max_num_hands=2) as tracker:
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("카메라 프레임을 읽지 못했습니다.")
                    return 1

                frame = cv2.flip(frame, 1)
                for hand in tracker.process(frame):
                    draw_hand(frame, hand)

                current_time = time.perf_counter()
                fps = 1.0 / max(current_time - previous_time, 1e-6)
                previous_time = current_time
                cv2.putText(frame, f"FPS {fps:.1f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("MediaPipe Hands - press Q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

