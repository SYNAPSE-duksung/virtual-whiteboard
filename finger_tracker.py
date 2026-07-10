import argparse
import csv
import math
import os
import time
from datetime import datetime

import cv2
import mediapipe as mp

# ----------------------------------------------------------------------------
# One Euro Filter (Casiez et al., 2012) - 실시간 좌표 스무딩용 저지연 필터
# ----------------------------------------------------------------------------
def _smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def _exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    """1차원 One Euro Filter. x, y 좌표에 각각 하나씩 사용."""

    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0:
            t_e = 1e-3  # 동일 타임스탬프 방지

        a_d = _smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = _exponential_smoothing(a_d, dx, self.dx_prev)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _smoothing_factor(t_e, cutoff)
        x_hat = _exponential_smoothing(a, x, self.x_prev)

        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


# ----------------------------------------------------------------------------
# MediaPipe landmark index 상수
# ----------------------------------------------------------------------------
WRIST = 0
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
# 지우기 제스처 판정용 (끝 landmark, PIP landmark) 쌍
FINGER_TIP_PIP_PAIRS = [(8, 6), (12, 10), (16, 14), (20, 18)]


def compute_pen_ratio(landmarks, w, h):
    """검지 Tip-DIP의 y좌표 차이를 손 크기(wrist-middle_mcp 거리)로 정규화.
    값이 작을수록 손가락이 눌린(pen-down) 상태로 판단."""
    tip = landmarks[INDEX_TIP]
    dip = landmarks[INDEX_DIP]
    wrist = landmarks[WRIST]
    mid_mcp = landmarks[MIDDLE_MCP]

    hand_size = math.hypot((wrist.x - mid_mcp.x) * w, (wrist.y - mid_mcp.y) * h)
    if hand_size < 1e-6:
        return 1.0

    y_diff = abs((tip.y - dip.y) * h)
    return y_diff / hand_size


def is_open_hand(landmarks):
    """4손가락 끝이 모두 PIP보다 위(y가 작음)에 있으면 손을 편 것으로 간주 (지우기 제스처)."""
    return all(landmarks[tip].y < landmarks[pip].y for tip, pip in FINGER_TIP_PIP_PAIRS)


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

    os.makedirs(args.output_dir, exist_ok=True)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"카메라(index={args.camera})를 열 수 없습니다.")

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("첫 프레임을 읽지 못했습니다. 카메라 연결을 확인하세요.")
    h, w = frame.shape[:2]
    canvas = None  # 필요 시 첫 프레임에서 초기화

    filter_x = filter_y = None  # 손이 처음 감지될 때 생성
    prev_point = None
    pen_down = False

    frame_id = 0
    csv_writer = None
    csv_file = None
    recording = not args.no_record

    def start_new_csv():
        nonlocal csv_writer, csv_file
        if csv_file is not None:
            csv_file.close()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(args.output_dir, f"coords_{ts}.csv")
        csv_file = open(path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ["frame_id", "timestamp", "hand_detected", "raw_x", "raw_y", "raw_z",
             "filtered_x", "filtered_y", "pen_ratio", "pen_down"]
        )
        print(f"[CSV] 기록 시작: {path}")
        return path

    if recording:
        start_new_csv()

    print("q: 종료 | r: 기록 on/off | c: 캔버스 지우기 | 손 펴기: 캔버스 지우기 제스처")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)  # 셀피뷰(거울모드)
            if canvas is None:
                canvas = frame.copy() * 0

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            now = time.time()
            hand_detected = False
            raw_x = raw_y = raw_z = None
            fx = fy = None
            pen_ratio = None

            if result.multi_hand_landmarks:
                hand_detected = True
                landmarks = result.multi_hand_landmarks[0].landmark
                tip = landmarks[INDEX_TIP]

                raw_x, raw_y, raw_z = tip.x * w, tip.y * h, tip.z

                if filter_x is None:
                    filter_x = OneEuroFilter(now, raw_x, min_cutoff=args.min_cutoff, beta=args.beta)
                    filter_y = OneEuroFilter(now, raw_y, min_cutoff=args.min_cutoff, beta=args.beta)
                    fx, fy = raw_x, raw_y
                else:
                    fx = filter_x(now, raw_x)
                    fy = filter_y(now, raw_y)

                pen_ratio = compute_pen_ratio(landmarks, w, h)
                # 히스테리시스: down 진입은 낮은 임계값, up 복귀는 높은 임계값
                if pen_down:
                    pen_down = pen_ratio < args.pen_up_thresh
                else:
                    pen_down = pen_ratio < args.pen_down_thresh

                point = (int(fx), int(fy))

                if is_open_hand(landmarks):
                    canvas[:] = 0
                    prev_point = None
                    pen_down = False
                else:
                    if pen_down and prev_point is not None:
                        cv2.line(canvas, prev_point, point, (0, 0, 255), 4, cv2.LINE_AA)
                    prev_point = point if pen_down else None

                # 시각화: 랜드마크 + pen 상태
                color = (0, 0, 255) if pen_down else (0, 255, 0)
                cv2.circle(frame, point, 8, color, -1)
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS
                )
            else:
                prev_point = None
                pen_down = False

            if recording and csv_writer is not None:
                csv_writer.writerow([
                    frame_id,
                    f"{now:.6f}",
                    int(hand_detected),
                    f"{raw_x:.2f}" if raw_x is not None else "",
                    f"{raw_y:.2f}" if raw_y is not None else "",
                    f"{raw_z:.5f}" if raw_z is not None else "",
                    f"{fx:.2f}" if fx is not None else "",
                    f"{fy:.2f}" if fy is not None else "",
                    f"{pen_ratio:.4f}" if pen_ratio is not None else "",
                    int(pen_down),
                ])
                if frame_id % 30 == 0:
                    csv_file.flush()

            frame_id += 1

            combined = cv2.addWeighted(frame, 1.0, canvas, 1.0, 0)
            status = f"REC" if recording else "PAUSED"
            cv2.putText(combined, f"[{status}] frame={frame_id}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow("SYNAPSE - Finger Tracker MVP", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                recording = not recording
                if recording:
                    start_new_csv()
                else:
                    print("[CSV] 기록 일시정지")
            elif key == ord("c"):
                canvas[:] = 0
                prev_point = None
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        if csv_file is not None:
            csv_file.close()
            print("[CSV] 파일 저장 완료")


if __name__ == "__main__":
    main()
