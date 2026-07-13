"""검지(8번) 실시간 좌표 추적 + One Euro 필터 + 자동 pen 판정 + CSV 수집 (D파트 학습 데이터).

CV 로직은 core 모듈(`HandTracker`, `FingertipSmoother`, `PenStateDetector`,
`compute_pen_ratio`, `is_open_hand`)을 재사용하고, 이 스크립트는 카메라 루프·시각화·CSV
기록 등 애플리케이션 관심사만 담당한다. 판정 로직은 controller/main.py와 동일한 구현을 공유.

실행: `python -m core.finger_tracker` (옵션: --camera, --min-cutoff, --beta,
--pen-down-thresh, --pen-up-thresh, --output-dir, --no-record)
"""

import argparse
import time

import cv2

from core.filters import FingertipSmoother
from core.pen_state import INDEX_TIP, PenStateDetector, compute_pen_ratio, is_open_hand
from core.recorder import CoordRecorder, CoordSample
from core.tracker import HandTracker


def main():
    parser = argparse.ArgumentParser(description="검지(8번) 실시간 좌표 추적 + 필터 + 드로잉 + CSV 추출")
    parser.add_argument("--camera", type=int, default=0, help="웹캠 인덱스 (기본 0)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--min-cutoff", type=float, default=1.0, help="One Euro Filter min_cutoff (낮을수록 부드러움)")
    parser.add_argument("--beta", type=float, default=0.3, help="One Euro Filter beta (높을수록 빠른 움직임에 민감)")
    parser.add_argument("--pen-down-thresh", type=float, default=0.55, help="pen-down 진입 임계값 (이 값보다 작으면 down)")
    parser.add_argument("--pen-up-thresh", type=float, default=0.70, help="pen-up 복귀 임계값 (히스테리시스, 이 값보다 크면 up)")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--no-record", action="store_true", help="시작 시 CSV 기록을 켜지 않음 ('r'로 직접 시작)")
    args = parser.parse_args()

    tracker = HandTracker(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6)
    smoother = FingertipSmoother(min_cutoff=args.min_cutoff, beta=args.beta)
    pen_state = PenStateDetector(down_thresh=args.pen_down_thresh, up_thresh=args.pen_up_thresh)
    recorder = CoordRecorder(args.output_dir)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"카메라(index={args.camera})를 열 수 없습니다.")

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("첫 프레임을 읽지 못했습니다. 카메라 연결을 확인하세요.")
    canvas = None  # 필요 시 첫 프레임에서 초기화

    prev_point = None
    frame_count = 0

    if not args.no_record:
        print(f"[CSV] 기록 시작: {recorder.start()}")

    print("q: 종료 | r: 기록 on/off | c: 캔버스 지우기 | 손 펴기: 캔버스 지우기 제스처")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)  # 셀피뷰(거울모드)
            if canvas is None:
                canvas = frame.copy() * 0

            h, w = frame.shape[:2]
            hands = tracker.process(frame)

            now = time.time()
            hand_detected = False
            raw_x = raw_y = raw_z = None
            fx = fy = None
            pen_ratio = None
            pen_down = False

            if hands:
                hand_detected = True
                lm = hands[0].normalized_landmarks
                raw_x, raw_y, raw_z = lm[INDEX_TIP][0] * w, lm[INDEX_TIP][1] * h, lm[INDEX_TIP][2]

                fx, fy = smoother.update(now, raw_x, raw_y)
                pen_ratio = compute_pen_ratio(lm, w, h)
                point = (int(fx), int(fy))

                if is_open_hand(lm):
                    canvas[:] = 0
                    prev_point = None
                    pen_state.reset()
                    pen_down = False
                else:
                    pen_down = pen_state.update(pen_ratio)
                    if pen_down and prev_point is not None:
                        cv2.line(canvas, prev_point, point, (0, 0, 255), 4, cv2.LINE_AA)
                    prev_point = point if pen_down else None

                # 시각화: 랜드마크 점 + pen 상태
                color = (0, 0, 255) if pen_down else (0, 255, 0)
                cv2.circle(frame, point, 8, color, -1)
                for px, py in hands[0].pixel_landmarks:
                    cv2.circle(frame, (int(px), int(py)), 2, (0, 255, 0), -1)
            else:
                smoother.reset()
                pen_state.reset()
                prev_point = None
                pen_down = False

            recorder.write(
                CoordSample(
                    hand_detected=hand_detected,
                    raw_x=raw_x,
                    raw_y=raw_y,
                    raw_z=raw_z,
                    filtered_x=fx,
                    filtered_y=fy,
                    pen_ratio=pen_ratio,
                    pen_down=pen_down,
                ),
                timestamp=now,
            )
            frame_count += 1

            combined = cv2.addWeighted(frame, 1.0, canvas, 1.0, 0)
            status = "REC" if recorder.recording else "PAUSED"
            cv2.putText(combined, f"[{status}] frame={frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow("SYNAPSE - Finger Tracker MVP", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                if recorder.toggle():
                    print(f"[CSV] 기록 시작: {recorder.path}")
                else:
                    print("[CSV] 기록 일시정지")
            elif key == ord("c"):
                canvas[:] = 0
                prev_point = None
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        if recorder.recording:
            recorder.close()
            print("[CSV] 파일 저장 완료")


if __name__ == "__main__":
    main()
