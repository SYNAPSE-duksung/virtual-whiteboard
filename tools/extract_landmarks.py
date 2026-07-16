"""녹화된 사선(main) 영상에서 오프라인으로 랜드마크를 추출해 학습용 CSV를 만든다.

``tools/record_dual.py``는 녹화 중 프레임 드랍을 막기 위해 영상만 저장하고 랜드마크는
추출하지 않는다. 이 스크립트는 저장된 ``*_main.mp4``를 나중에 다시 읽어 프레임 단위로
아래 두 CSV를 같은 패스에서 만든다.

1. ``{base}_coords.csv`` — 손끝(검지, 8번) 좌표·pen_ratio·pen_down 등 기존 판정용 스키마
   (``CSV_COLUMNS + video_id``). A의 라이브 수집 CSV(``core.recorder``가 그대로 쓰는 순정
   ``CSV_COLUMNS``)와는 마지막의 ``video_id`` 컬럼 하나만 다르며, 값은 영상 파일 base명
   (예: ``minjin_bright_slow_01``)이다.
2. ``{base}_landmarks.csv`` — 손 랜드마크 21개 전체의 MediaPipe 정규화 좌표
   (``frame_id, timestamp, hand_detected, x0, y0, z0, …, x20, y20, z20, video_id``,
   총 3+63+1=67열). D파트가 각도 변화·속도·저크 등 kinematic feature를 손끝 한 점이
   아닌 손 전체 골격으로 계산할 수 있도록 하기 위함이다. 값은 mediapipe가 반환하는
   정규화 좌표(0~1 범위) 그대로 ``.6f``로 기록하며, coords 쪽과 달리 픽셀 변환이나
   스무딩을 거치지 않은 raw 값이다.

두 파일 모두 같은 폴더의 ``{base}_frames.csv``(측면 카메라 라벨과의 프레임 정렬 기준)가
있으면 그 타임스탬프를 그대로 쓰고, 없으면 영상 FPS로부터 근사한다. D파트가 여러 영상의
행을 한 CSV로 합쳐도 ``video_id``로 클립 경계를 구분해 시계열 feature를 영상 내에서만
계산하고, 클립 단위로 train/test를 분리할 수 있도록 한다.

내부적으로는 mediapipe를 두 번 돌리지 않기 위해 ``core.PenTracker`` 대신
``HandTracker`` + ``FingertipSmoother`` + ``PenStateDetector`` 부품을 직접 조합해
``PenTracker.process()``와 동일한 로직으로 pen_down/erase_gesture를 재현하면서, 동시에
그 프레임의 21개 정규화 랜드마크를 landmarks CSV에 기록한다 (``core.finger_tracker``가
쓰는 것과 동일한 공인 패턴).

전제: ``core/tracker.py``, ``core/filters.py``, ``core/pen_state.py``,
``core/pen_tracker.py``, ``core/recorder.py``는 #8 브랜치(core 리팩토링)에서 도입됐다.
머지 전에는 안내 메시지와 함께 종료한다.

사용 예시:
  python -m tools.extract_landmarks
      data/dataset/recordings/ 아래 모든 *_main.mp4 일괄 처리 (이미 _coords.csv와
      _landmarks.csv가 모두 있는 영상은 건너뜀)
  python -m tools.extract_landmarks data/dataset/recordings/minjin_bright_slow_01_main.mp4
      개별 파일 지정
  python -m tools.extract_landmarks data/dataset/recordings --overwrite
      폴더 지정 + 기존 결과 덮어쓰기
  python -m tools.extract_landmarks --min-cutoff 1.5 --pen-down-thresh 0.5
      필터/판정 파라미터 조정 후 재추출
  python -m tools.extract_landmarks --min-detection-confidence 0.5 --overwrite
      손 검출률이 낮을 때 검출 신뢰도 임계값을 낮춰 재추출
      (녹화 전 각도 점검은 record_dual의 "검출 테스트" 버튼 사용)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

try:
    from core.filters import FingertipSmoother
    from core.pen_state import INDEX_TIP, PenStateDetector, compute_pen_ratio, is_open_hand
    from core.pen_tracker import PenFrame
    from core.recorder import CSV_COLUMNS, CoordSample
    from core.tracker import HandTracker
except ImportError as exc:
    raise SystemExit(
        "core 모듈을 가져올 수 없습니다.\n"
        "  - #8 브랜치(core 리팩토링) 머지 후 `git pull`이 필요합니다.\n"
        "  - mediapipe==0.10.21이 설치되어 있는지 확인하세요: "
        "`python -m pip install -r requirements.txt`\n"
        f"(원본 오류: {exc})"
    )

DEFAULT_INPUT_DIR = Path("data/dataset/recordings")
_DEFAULT_FPS = 30.0
_NUM_LANDMARKS = 21

LANDMARK_CSV_COLUMNS = (
    ["frame_id", "timestamp", "hand_detected"]
    + [f"{axis}{i}" for i in range(_NUM_LANDMARKS) for axis in ("x", "y", "z")]
    + ["video_id"]
)


def _fmt(value: float | None, spec: str) -> str:
    return format(value, spec) if value is not None else ""


def _coord_row(frame_id: int, timestamp: float, sample: CoordSample, video_id: str) -> list[object]:
    """``CoordRecorder.write()``와 자릿수까지 동일한 한 행 + ``video_id``를 만든다."""
    return [
        frame_id,
        f"{timestamp:.6f}",
        int(sample.hand_detected),
        _fmt(sample.raw_x, ".2f"),
        _fmt(sample.raw_y, ".2f"),
        _fmt(sample.raw_z, ".5f"),
        _fmt(sample.filtered_x, ".2f"),
        _fmt(sample.filtered_y, ".2f"),
        _fmt(sample.pen_ratio, ".4f"),
        int(sample.pen_down),
        video_id,
    ]


def _landmarks_row(
    frame_id: int,
    timestamp: float,
    hand_detected: bool,
    normalized_landmarks,
    video_id: str,
) -> list[object]:
    """랜드마크 21개(x,y,z, 정규화 좌표) 한 행 + ``video_id``를 만든다.

    ``normalized_landmarks``는 ``HandTracker``가 반환하는 (21, 3) 배열(mediapipe
    정규화 좌표, 0~1 범위)이며, 손 미검출 프레임에서는 ``None``을 받아 63개 빈 문자열을 채운다.
    """
    row: list[object] = [frame_id, f"{timestamp:.6f}", int(hand_detected)]
    if normalized_landmarks is None:
        row.extend([""] * (_NUM_LANDMARKS * 3))
    else:
        for i in range(_NUM_LANDMARKS):
            x, y, z = normalized_landmarks[i]
            row.extend([f"{float(x):.6f}", f"{float(y):.6f}", f"{float(z):.6f}"])
    row.append(video_id)
    return row


def _load_frame_timestamps(frames_csv: Path) -> dict[int, float]:
    """``{base}_frames.csv``에서 frame_index -> elapsed_sec 매핑을 읽는다.

    구형(``frame_index,elapsed_sec``)·신형(``frame_index,elapsed_sec,main_ok,side_ok``)
    헤더를 모두 지원한다.
    """
    mapping: dict[int, float] = {}
    with open(frames_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                mapping[int(row["frame_index"])] = float(row["elapsed_sec"])
            except (KeyError, ValueError, TypeError):
                continue
    return mapping


def _find_main_videos(paths: list[str]) -> list[Path]:
    """인자로 받은 파일/폴더 목록에서 ``*_main.mp4`` 경로들을 모은다."""
    if not paths:
        paths = [str(DEFAULT_INPUT_DIR)]

    videos: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            videos.extend(sorted(p.glob("*_main.mp4")))
        elif p.is_file():
            videos.append(p)
        else:
            print(f"[건너뜀] 경로를 찾을 수 없습니다: {p}")
    return videos


def _video_base(video_path: Path) -> str:
    """``*_main.mp4``에서 ``_main.mp4``를 뗀 base명을 계산한다 (``video_id`` 값으로도 재사용)."""
    name = video_path.name
    if name.endswith("_main.mp4"):
        return name[: -len("_main.mp4")]
    return video_path.stem


def _output_path(video_path: Path) -> Path:
    return video_path.with_name(f"{_video_base(video_path)}_coords.csv")


def _landmarks_output_path(video_path: Path) -> Path:
    return video_path.with_name(f"{_video_base(video_path)}_landmarks.csv")


def _frames_csv_path(video_path: Path) -> Path:
    return video_path.with_name(f"{_video_base(video_path)}_frames.csv")


def process_video(video_path: Path, *, args: argparse.Namespace) -> None:
    out_path = _output_path(video_path)
    landmarks_path = _landmarks_output_path(video_path)
    if out_path.exists() and landmarks_path.exists() and not args.overwrite:
        print(
            f"[건너뜀] 이미 존재함: {out_path.name}, {landmarks_path.name} "
            "(--overwrite로 재생성 가능)"
        )
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[건너뜀] 영상을 열 수 없습니다: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = _DEFAULT_FPS

    frames_csv = _frames_csv_path(video_path)
    timestamps = _load_frame_timestamps(frames_csv) if frames_csv.exists() else None
    video_id = _video_base(video_path)

    # mediapipe를 두 번 돌리지 않기 위해 core.PenTracker를 쓰지 않고 부품(HandTracker +
    # FingertipSmoother + PenStateDetector)을 직접 조합한다. 21개 랜드마크가 필요한데
    # PenTracker.process()의 반환값(PenFrame)에는 검지 끝 좌표만 담겨 있기 때문이다.
    tracker = HandTracker(
        max_num_hands=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    smoother = FingertipSmoother(min_cutoff=args.min_cutoff, beta=args.beta, d_cutoff=1.0)
    pen_state = PenStateDetector(down_thresh=args.pen_down_thresh, up_thresh=args.pen_up_thresh)

    total_frames = 0
    hand_detected_frames = 0
    pen_down_frames = 0

    try:
        with open(out_path, "w", newline="", encoding="utf-8") as f_coords, open(
            landmarks_path, "w", newline="", encoding="utf-8"
        ) as f_landmarks:
            coords_writer = csv.writer(f_coords)
            coords_writer.writerow(CSV_COLUMNS + ["video_id"])
            landmarks_writer = csv.writer(f_landmarks)
            landmarks_writer.writerow(LANDMARK_CSV_COLUMNS)

            frame_id = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if timestamps is not None and frame_id in timestamps:
                    timestamp = timestamps[frame_id]
                else:
                    timestamp = frame_id / fps

                # 아래 블록은 core.PenTracker.process()와 동작 동기화 유지 —
                # 로직을 바꿀 경우 core/pen_tracker.py의 PenTracker.process()도 함께 확인할 것.
                hands = tracker.process(frame)
                if not hands:
                    smoother.reset()
                    pen_state.reset()
                    pen_frame = PenFrame.empty()
                    normalized_landmarks = None
                else:
                    hand = hands[0]
                    height, width = frame.shape[:2]
                    landmarks = hand.normalized_landmarks

                    raw_x = float(landmarks[INDEX_TIP][0]) * width
                    raw_y = float(landmarks[INDEX_TIP][1]) * height
                    raw_z = float(landmarks[INDEX_TIP][2])
                    smooth_x, smooth_y = smoother.update(timestamp, raw_x, raw_y)
                    fingertip = (int(round(smooth_x)), int(round(smooth_y)))

                    pen_ratio = compute_pen_ratio(landmarks, width, height)
                    erase_gesture = is_open_hand(landmarks)
                    if erase_gesture:
                        # 지우기 제스처 중에는 펜을 내리지 않는다.
                        pen_state.reset()
                        pen_down = False
                    else:
                        pen_down = pen_state.update(pen_ratio)

                    pen_frame = PenFrame(
                        hand_detected=True,
                        fingertip=fingertip,
                        raw_fingertip=(raw_x, raw_y),
                        raw_z=raw_z,
                        pen_down=pen_down,
                        erase_gesture=erase_gesture,
                        pen_ratio=pen_ratio,
                    )
                    normalized_landmarks = landmarks

                sample = CoordSample.from_pen_frame(pen_frame)
                coords_writer.writerow(_coord_row(frame_id, timestamp, sample, video_id))
                landmarks_writer.writerow(
                    _landmarks_row(
                        frame_id, timestamp, pen_frame.hand_detected, normalized_landmarks, video_id
                    )
                )

                total_frames += 1
                if sample.hand_detected:
                    hand_detected_frames += 1
                if sample.pen_down:
                    pen_down_frames += 1

                frame_id += 1
    finally:
        cap.release()
        tracker.close()

    if timestamps is None:
        print(f"  (참고: {frames_csv.name} 없음 — fps={fps:.2f} 기반으로 timestamp 근사)")

    detect_rate = (hand_detected_frames / total_frames * 100) if total_frames else 0.0
    pen_down_rate = (pen_down_frames / total_frames * 100) if total_frames else 0.0
    print(
        f"[완료] {video_path.name} -> {out_path.name}, {landmarks_path.name} | "
        f"프레임 {total_frames} | 손 검출률 {detect_rate:.1f}% | pen-down 비율 {pen_down_rate:.1f}%"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "paths",
        nargs="*",
        help="개별 *_main.mp4 파일 또는 폴더 경로 (기본: data/dataset/recordings/ 전체)",
    )
    parser.add_argument("--min-cutoff", type=float, default=1.0, help="One Euro Filter min_cutoff")
    parser.add_argument("--beta", type=float, default=0.3, help="One Euro Filter beta")
    parser.add_argument("--pen-down-thresh", type=float, default=0.55, help="pen-down 진입 임계값")
    parser.add_argument("--pen-up-thresh", type=float, default=0.70, help="pen-up 복귀 임계값 (히스테리시스)")
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.7,
        help="MediaPipe 손 검출 신뢰도 임계값 (기본 0.7 — 검출률이 낮으면 0.5 정도로 낮춰 재추출)",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.6,
        help="MediaPipe 손 추적 신뢰도 임계값 (기본 0.6)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="기존 _coords.csv/_landmarks.csv가 있어도 재생성"
    )
    args = parser.parse_args()

    videos = _find_main_videos(args.paths)
    if not videos:
        print("처리할 *_main.mp4 파일이 없습니다.")
        return 1

    for video_path in videos:
        process_video(video_path, args=args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
