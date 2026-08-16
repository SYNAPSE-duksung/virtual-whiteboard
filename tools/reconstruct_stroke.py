"""메인(노트북) 궤적 + 측면(폰) pen up/down 신호를 결합해 손글씨를 복원한다 — 듀얼 카메라
계획(CLAUDE.md "듀얼 카메라 기반 pen up/down 개선 + 글자 복원 실험 계획") 1단계의 마지막 단계.

메인 카메라는 필기 평면을 정면으로 보므로 궤적(정류 좌표) 복원에 강하고, 측면 카메라는
손끝-책상 간격이 왜곡 없이 보이므로 pen up/down 판정에 강하다는 전제로, 각 카메라가 잘하는
일만 시킨 뒤 타임스탬프로 결합한다. **ML/DL 없이 두 스트림만으로 글자가 복원되는지**를
먼저 판단하는 것이 목적이며, 여기서 복원이 안 되면(CLAUDE.md 2단계) 비로소 학습 기반 결합을
검토한다.

전제 조건
---------
1. ``tools/record_dual.py``로 메인(``{base}_main.mp4``)·측면(``{base}_side.mp4``)·
   싱크(``{base}_frames.csv``)를 함께 녹화해 둘 것.
2. 메인 영상은 ``python -m tools.extract_landmarks --calibration output/calibration.json``
   으로 **정류 좌표(rectified_x/y) 포함** ``{base}_coords.csv``를 미리 추출해 둘 것.
3. 측면 카메라의 "책상면 기준선"을 지정할 것 — 처음 실행 시 측면 영상 첫 프레임에서
   마우스로 2점(책상 가장자리 위의 두 점)을 클릭하면 되고, ``--save-baseline``으로
   저장해두면 같은 카메라 구도에서 재사용할 수 있다(``core.side_contact.SideBaseline``).

측면 영상에서 손끝-기준선 거리(px)에 히스테리시스(``core.side_contact.SideContactDetector``)
를 적용해 자동 pen up/down을 얻고, 메인 CSV의 각 행(타임스탬프)에 가장 가까운 측면 표본의
pen 상태를 매칭해 정류 좌표를 stroke로 잘라 ``ui.canvas.StrokeCanvas``에 렌더링한다.

사용 예시:
  python -m tools.reconstruct_stroke
      data/dataset/recordings/ 아래 모든 *_side.mp4 일괄 처리 (첫 영상에서 기준선을
      대화형으로 한 번 지정해 배치 전체에 재사용)
  python -m tools.reconstruct_stroke data/dataset/recordings/minjin_bright_slow_01_side.mp4
      개별 파일 지정
  python -m tools.reconstruct_stroke --baseline 40 500 900 480 --save-baseline
      기준선을 클릭 없이 직접 지정(헤드리스 환경) + 다음에 재사용하도록 저장
  python -m tools.reconstruct_stroke --ocr
      복원한 이미지를 EasyOCR로 인식해 텍스트를 함께 출력 (easyocr 설치 필요, 선택)

출력: ``{base}_reconstructed.png`` (흰 배경, 검은 잉크 — OCR 입력으로 바로 쓸 수 있는 포맷).
"""

from __future__ import annotations

import argparse
import bisect
import csv
from pathlib import Path

import cv2
import numpy as np

from core.geometry import DEFAULT_CALIBRATION_PATH, PerspectiveCalibration
from core.side_contact import (
    DEFAULT_BASELINE_PATH,
    DEFAULT_DOWN_PX,
    DEFAULT_UP_PX,
    SideBaseline,
    SideBaselinePicker,
    SideContactDetector,
    estimate_baseline,
    measure,
)
from core.tracker import HandTracker
from ui.canvas import StrokeCanvas

DEFAULT_INPUT_DIR = Path("data/dataset/recordings")
_DEFAULT_FPS = 30.0
_SIDE_SUFFIX = "_side.mp4"


# ----------------------------------------------------------------------
# 경로 탐색
# ----------------------------------------------------------------------

def _base_of(side_video: Path) -> str:
    name = side_video.name
    return name[: -len(_SIDE_SUFFIX)] if name.endswith(_SIDE_SUFFIX) else side_video.stem


def _resolve_side_video(raw: str) -> Path | None:
    p = Path(raw)
    if p.is_file() and p.name.endswith(_SIDE_SUFFIX):
        return p
    # base 이름만 준 경우: 기본 폴더에서 찾는다.
    candidate = DEFAULT_INPUT_DIR / f"{raw}{_SIDE_SUFFIX}"
    if candidate.is_file():
        return candidate
    # 다른 산출물(_main.mp4/_coords.csv/_frames.csv) 경로를 줬을 수도 있다.
    for suffix in ("_main.mp4", "_coords.csv", "_landmarks.csv", "_frames.csv"):
        if p.name.endswith(suffix):
            sibling = p.with_name(p.name[: -len(suffix)] + _SIDE_SUFFIX)
            if sibling.is_file():
                return sibling
    return None


def _find_side_videos(paths: list[str]) -> list[Path]:
    if not paths:
        paths = [str(DEFAULT_INPUT_DIR)]
    videos: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            videos.extend(sorted(p.glob(f"*{_SIDE_SUFFIX}")))
            continue
        resolved = _resolve_side_video(raw)
        if resolved is not None:
            videos.append(resolved)
        else:
            print(f"[건너뜀] 측면 영상을 찾을 수 없습니다: {raw}")
    return videos


# ----------------------------------------------------------------------
# 기준선(책상면) 대화형 지정
# ----------------------------------------------------------------------

def _interactive_baseline(side_video: Path) -> SideBaseline:
    """측면 영상 첫 프레임에서 마우스로 2점을 찍어 ``SideBaseline``을 만든다."""
    cap = cv2.VideoCapture(str(side_video))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"측면 영상에서 첫 프레임을 읽지 못했습니다: {side_video}")

    picker = SideBaselinePicker()
    # ASCII로만 작성 (Hershey 폰트에 한글 글리프가 없어 한글은 ?????로 깨진다 —
    # controller/main.py·tools/calibrate.py와 동일 규칙).
    window = "Side baseline - click 2 pts on desk edge (z undo, Enter/S confirm, Esc abort)"
    message = ""

    def on_mouse(event: int, x: int, y: int, flags: int, userdata: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            picker.add_point(x, y)

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    try:
        while True:
            display = frame.copy()
            for i, (x, y) in enumerate(picker.points):
                pt = (int(round(x)), int(round(y)))
                cv2.circle(display, pt, 6, (0, 215, 255), -1)
                cv2.putText(display, str(i + 1), (pt[0] + 8, pt[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2, cv2.LINE_AA)
            if len(picker.points) == 2:
                p1 = tuple(int(round(v)) for v in picker.points[0])
                p2 = tuple(int(round(v)) for v in picker.points[1])
                cv2.line(display, p1, p2, (60, 220, 60), 2, cv2.LINE_AA)

            guide = (
                f"next click: {picker.next_label} ({picker.count}/2)"
                if not picker.is_complete
                else "2 points done - Enter/S confirm / z undo / Esc abort"
            )
            cv2.putText(display, guide, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 2, cv2.LINE_AA)
            if message:
                cv2.putText(display, message, (15, display.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow(window, display)

            key = cv2.waitKey(20) & 0xFF
            if key == ord("z"):
                picker.undo()
                message = ""
            elif key in (13, ord("s")):
                baseline, err = picker.try_build()
                if baseline is None:
                    message = err or "click 2 points first."
                else:
                    return baseline
            elif key == 27:
                raise SystemExit("baseline calibration aborted.")
    finally:
        cv2.destroyWindow(window)


# ----------------------------------------------------------------------
# 측면 영상 처리
# ----------------------------------------------------------------------

def _load_side_timestamps(frames_csv: Path) -> list[float] | None:
    """``{base}_frames.csv``에서 ``side_ok==1``인 행만 순서대로 골라 elapsed_sec 목록을 만든다.

    ``record_dual.py``는 프레임 루프 한 틱마다 frames.csv에 한 행을 쓰고, 그 틱에 실제로
    측면 프레임을 받았을 때만(``side_ok=1``) ``_side.mp4``에도 한 프레임을 쓴다. 따라서
    ``_side.mp4``의 N번째(0-index) 프레임에 대응하는 시각은, frames.csv에서
    ``side_ok==1``인 행을 순서대로 센 N번째 행의 ``elapsed_sec``다 — ``frame_index``를
    그대로 쓰면(메인/측면 드랍이 있는 경우) 어긋난다.
    """
    if not frames_csv.exists():
        return None
    timestamps: list[float] = []
    with open(frames_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # 구형(2컬럼) frames.csv에는 side_ok가 없다 — 드랍이 기록 안 된 것으로 보고
                # 모든 행을 side 프레임으로 취급한다(기존 _load_frame_timestamps와 동일 완화).
                if int(row.get("side_ok", 1)):
                    timestamps.append(float(row["elapsed_sec"]))
            except (KeyError, ValueError, TypeError):
                continue
    return timestamps


def _process_side_video(
    side_video: Path,
    baseline: SideBaseline,
    frames_csv: Path,
    *,
    down_px: float,
    up_px: float,
    min_detection_confidence: float,
    min_tracking_confidence: float,
) -> tuple[list[float], list[bool], int, int]:
    """측면 영상을 훑어 (timestamp 오름차순, pen_down) 표본과 검출 통계를 만든다."""
    side_ts = _load_side_timestamps(frames_csv)

    cap = cv2.VideoCapture(str(side_video))
    if not cap.isOpened():
        raise SystemExit(f"측면 영상을 열 수 없습니다: {side_video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = _DEFAULT_FPS

    tracker = HandTracker(
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    detector = SideContactDetector(down_px=down_px, up_px=up_px)

    timestamps: list[float] = []
    pen_states: list[bool] = []
    total = 0
    detected = 0
    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts = side_ts[idx] if side_ts is not None and idx < len(side_ts) else idx / fps

            hands = tracker.process(frame)
            if not hands:
                detector.reset()
                pen_down = False
            else:
                distance = measure(hands[0], baseline)
                pen_down = detector.update(distance)
                detected += 1

            timestamps.append(ts)
            pen_states.append(pen_down)
            total += 1
            idx += 1
    finally:
        cap.release()
        tracker.close()

    if side_ts is None:
        print(f"  (참고: {frames_csv.name} 없음 — fps={fps:.2f} 기반으로 timestamp 근사)")
    elif len(side_ts) != total:
        print(f"  (주의: frames.csv의 side 프레임 수({len(side_ts)})와 실제 영상 프레임 수"
              f"({total})가 다릅니다 — 넘치는 구간은 fps 근사로 대체됩니다)")

    return timestamps, pen_states, total, detected


def _nearest_pen_down(t: float, side_timestamps: list[float], side_pen: list[bool]) -> bool:
    """메인 프레임 타임스탬프 ``t``와 가장 가까운 측면 표본의 pen_down을 찾는다."""
    if not side_timestamps:
        return False
    i = bisect.bisect_left(side_timestamps, t)
    if i == 0:
        return side_pen[0]
    if i == len(side_timestamps):
        return side_pen[-1]
    before, after = side_timestamps[i - 1], side_timestamps[i]
    return side_pen[i - 1] if (t - before) <= (after - t) else side_pen[i]


# ----------------------------------------------------------------------
# 메인 CSV 로드 + stroke 재구성
# ----------------------------------------------------------------------

def _load_coords(coords_csv: Path) -> list[dict] | None:
    """정류 좌표 컬럼이 있는지 확인하며 ``{base}_coords.csv``를 읽는다."""
    with open(coords_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "rectified_x" not in fieldnames or "rectified_y" not in fieldnames:
            print(
                f"[건너뜀] {coords_csv.name}에 rectified_x/rectified_y 컬럼이 없습니다 — "
                "`python -m tools.extract_landmarks --calibration ... --overwrite`로 "
                "먼저 정류 좌표를 포함해 재추출하세요."
            )
            return None
        return list(reader)


def _reconstruct(
    coords_rows: list[dict],
    side_timestamps: list[float],
    side_pen: list[bool],
    dst_size: tuple[int, int],
    *,
    line_color: tuple[int, int, int],
    line_thickness: int,
) -> tuple[np.ndarray, int]:
    """메인 정류 좌표를 측면 pen 신호로 잘라 ``StrokeCanvas``에 그린다."""
    canvas = StrokeCanvas(dst_size[0], dst_size[1], line_color=line_color, line_thickness=line_thickness)
    stroke_count = 0
    for row in coords_rows:
        hand_detected = row.get("hand_detected") == "1"
        rx, ry = row.get("rectified_x", ""), row.get("rectified_y", "")
        pen_down = False
        if hand_detected and rx != "" and ry != "":
            t = float(row["timestamp"])
            pen_down = _nearest_pen_down(t, side_timestamps, side_pen)

        if pen_down:
            point = (int(round(float(rx))), int(round(float(ry))))
            if canvas.is_pen_down:
                canvas.move(point)
            else:
                canvas.pen_down(point)
                stroke_count += 1
        else:
            canvas.pen_up()
    return canvas.image, stroke_count


def _run_ocr(image: np.ndarray) -> None:
    """복원 결과를 EasyOCR로 인식한다 (선택 기능).

    ``ai/ocr.py``는 모듈 최상단에서 ``easyocr.Reader(...)``를 생성해(무거운 모델 로드)
    import 시점 부작용이 있으므로 재사용하지 않고, 이 함수 안에서만 지연 import한다 —
    ``--ocr``을 주지 않으면 이 도구의 기본 실행 경로에는 easyocr 의존성이 전혀 없다.
    """
    try:
        import easyocr
    except ImportError:
        print("[OCR] easyocr가 설치되어 있지 않습니다 — `pip install easyocr` 후 다시 시도하세요.")
        return
    reader = easyocr.Reader(["ko", "en"])
    results = reader.readtext(image)
    if not results:
        print("[OCR] 인식된 텍스트가 없습니다.")
        return
    print("[OCR] 인식 결과:")
    for _, text, confidence in results:
        print(f"  {text!r} (신뢰도 {confidence:.2f})")


# ----------------------------------------------------------------------
# 배치 처리
# ----------------------------------------------------------------------

def process_base(
    side_video: Path,
    *,
    calibration: PerspectiveCalibration,
    baseline: SideBaseline,
    args: argparse.Namespace,
) -> None:
    base = _base_of(side_video)
    coords_csv = side_video.with_name(f"{base}_coords.csv")
    frames_csv = side_video.with_name(f"{base}_frames.csv")
    out_path = side_video.with_name(f"{base}_reconstructed.png")

    if not coords_csv.is_file():
        print(
            f"[건너뜀] {coords_csv.name}이 없습니다 — 먼저 "
            f"`python -m tools.extract_landmarks --calibration ... {base}_main.mp4`로 추출하세요."
        )
        return
    if out_path.exists() and not args.overwrite:
        print(f"[건너뜀] 이미 존재함: {out_path.name} (--overwrite로 재생성 가능)")
        return

    coords_rows = _load_coords(coords_csv)
    if coords_rows is None:
        return

    side_ts, side_pen, side_total, side_detected = _process_side_video(
        side_video, baseline, frames_csv,
        down_px=args.down_px, up_px=args.up_px,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    image, stroke_count = _reconstruct(
        coords_rows, side_ts, side_pen, calibration.dst_size,
        line_color=(0, 0, 0), line_thickness=args.line_thickness,
    )
    cv2.imwrite(str(out_path), image)

    side_detect_rate = (side_detected / side_total * 100) if side_total else 0.0
    side_down_rate = (sum(side_pen) / side_total * 100) if side_total else 0.0
    print(
        f"[완료] {base} -> {out_path.name} | 메인 {len(coords_rows)}행 | "
        f"측면 {side_total}프레임(검출률 {side_detect_rate:.1f}%, pen-down {side_down_rate:.1f}%) | "
        f"stroke {stroke_count}개"
    )

    if args.ocr:
        _run_ocr(image)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "paths",
        nargs="*",
        help="개별 *_side.mp4/base명/기타 산출물 경로 또는 폴더 (기본: data/dataset/recordings/ 전체)",
    )
    parser.add_argument(
        "--calibration", type=str, default=str(DEFAULT_CALIBRATION_PATH),
        help="메인 궤적 추출에 쓴 4점 캘리브레이션 — dst_size를 복원 캔버스 크기로 사용",
    )
    parser.add_argument("--baseline-json", type=str, default=str(DEFAULT_BASELINE_PATH))
    parser.add_argument(
        "--baseline", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"), default=None,
        help="측면 기준선을 직접 지정 (대화형 클릭 대신 사용, headless 환경용)",
    )
    parser.add_argument(
        "--save-baseline", action="store_true",
        help="이번에 지정(대화형 또는 --baseline)한 기준선을 --baseline-json에 저장",
    )
    parser.add_argument(
        "--recalibrate-baseline", action="store_true",
        help="저장된 기준선 파일이 있어도 다시 대화형으로 지정",
    )
    parser.add_argument("--down-px", type=float, default=DEFAULT_DOWN_PX, help="접촉 진입 문턱(px)")
    parser.add_argument("--up-px", type=float, default=DEFAULT_UP_PX, help="접촉 해제 문턱(px, 히스테리시스)")
    parser.add_argument("--min-detection-confidence", type=float, default=0.7)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.6)
    parser.add_argument("--line-thickness", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true", help="기존 _reconstructed.png가 있어도 재생성")
    parser.add_argument(
        "--ocr", action="store_true",
        help="복원 결과를 EasyOCR로 인식해 텍스트를 함께 출력 (선택, easyocr 설치 필요)",
    )
    args = parser.parse_args()

    try:
        calibration = PerspectiveCalibration.load(args.calibration)
    except (FileNotFoundError, OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(
            f"캘리브레이션을 읽을 수 없습니다 ({args.calibration}): {exc}\n"
            "먼저 `python -m tools.calibrate` 또는 `python -m controller.main`의 K 키로 4점을 지정하세요."
        )

    side_videos = _find_side_videos(args.paths)
    if not side_videos:
        print("처리할 *_side.mp4 파일이 없습니다.")
        return 1

    if args.baseline is not None:
        baseline = estimate_baseline((args.baseline[0], args.baseline[1]), (args.baseline[2], args.baseline[3]))
    elif Path(args.baseline_json).is_file() and not args.recalibrate_baseline:
        baseline = SideBaseline.load(args.baseline_json)
        print(f"[기준선] 저장된 파일 사용: {args.baseline_json}")
    else:
        print(
            f"[기준선] {side_videos[0].name}의 첫 프레임으로 대화형 지정을 시작합니다 "
            "(이 배치의 모든 영상에 동일하게 적용됩니다 — 카메라를 옮겼다면 개별 실행하세요)."
        )
        baseline = _interactive_baseline(side_videos[0])

    if args.save_baseline or args.baseline is not None:
        saved = baseline.save(args.baseline_json)
        print(f"[기준선] 저장: {saved}")

    for side_video in side_videos:
        process_base(side_video, calibration=calibration, baseline=baseline, args=args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
