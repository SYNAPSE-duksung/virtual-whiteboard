"""메인/측면 영상의 손 움직임을 상호상관(cross-correlation)해 카메라 간 지연(오프셋)을
자동 추정한다 — ``label_frames.py``의 "지연 오프셋(프레임)" 스핀박스를 눈으로 맞추는 대신
초기값을 채우거나 그 값을 검산하는 도구.

배경
----
``label_frames.py``는 두 카메라(메인·측면)를 시간 정렬하기 위해 라벨러가 스핀박스를 손으로
조절해야 한다. 원래 설계는 촬영 시작 시 박수 같은 동기화 마커를 두 화면에서 각각 눈으로
확인해 맞추는 방식이다. 마커 없이(예: 첫 toggle을 마커로 쓰려는 시도) 진행하면 구조적인
문제가 생긴다 — **pen-down 순간(toggle)은 측면 영상에서만 명확하다.** 메인 카메라 각도
(사선/거의 수직)에서는 손끝이 책상에 닿는 순간을 눈으로 확정하기 어렵다는 게 애초에 이
프로젝트가 측면 카메라를 쓰는 이유이기 때문에, "측면에서 본 toggle 프레임"에 대응하는
"메인에서 본 같은 순간"을 눈으로 찾을 수가 없다.

이 도구는 마커에 기대지 않는다 — 촬영 내내 자연스럽게 일어나는 **손의 전체적인 움직임
자체**를 동기화 신호로 쓴다. 메인·측면 각 영상에 독립적인 ``HandTracker``를 돌려(같은
MediaPipe 인스턴스를 두 스트림에 번갈아 먹이면 내부 트래킹 상태가 섞이므로 반드시
분리 — ``core.side_contact``·``core.pen_tracker``의 측면 카메라 처리와 같은 이유) 검지 끝의
프레임별 움직임 크기(정규화 좌표 기준 변위)를 시계열로 뽑고, 두 시계열을 상호상관해
가장 잘 겹치는 시차를 찾는다. 마이크 2개로 소리 도착 시간차를 구하는 것과 같은 원리다.

⚠️ 한계 (반드시 읽을 것)
------------------------
- 손의 전체적인 움직임 패턴이 두 카메라에서 상관돼 있다는 가정에 의존한다. 손이 오래
  멈춰 있거나 움직임이 단조로우면 신호가 밋밋해져 추정이 부정확해질 수 있다 — 출력되는
  **정규화 상관계수(-1~1, 코사인 유사도)** 가 낮으면(기준: 0.3 미만) 신뢰하지 말 것.
- 검색 범위는 기본 ±2.0초(``--max-lag-sec``)다. 실제 지연이 이보다 크면 못 잡는다(경고 출력).
- 이 값은 ``label_frames.py``의 "지연 오프셋(프레임)"과 **정확히 같은 정의**
  (``g = side_map[k] - offset``, ``k``=측면 영상 내부 프레임 번호, ``g``=보정된 전역
  frame_index)로 계산된다. ``--save``를 주면 ``labeling_offsets.json``에 바로 기록되어
  ``label_frames.py``를 열 때 자동 복원된다.
- 그래도 **처음 몇 개 영상은 label_frames.py에서 육안으로 한 번씩 검산**하길 권한다.
  같은 촬영 세션·같은 물리적 연결(폰 위치, 케이블/Iriun 설정 안 바꿈)이면 오프셋을 세션
  전체에 재사용해도 된다는 게 이 프로젝트의 기존 가정이지만, 그 가정 자체도 처음 몇 개는
  교차 확인하는 게 안전하다.

사용 예시:
  python -m tools.estimate_camera_offset minjin_bright_slow_01
      단일 base 처리, 결과만 출력(파일에 쓰지 않음)
  python -m tools.estimate_camera_offset minjin_bright_slow_01 --save
      결과를 output/../labeling_offsets.json에 기록 (스핀박스 범위 -30~30으로 클램프)
  python -m tools.estimate_camera_offset --plot
      data/dataset/recordings/ 전체 배치 처리 + 정렬된 신호·상관계수 그래프 PNG 저장
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from core.pen_state import INDEX_TIP
from core.tracker import HandTracker
from tools.label_frames import (
    FRAMES_SUFFIX,
    MAIN_SUFFIX,
    OFFSETS_FILENAME,
    OUTPUT_DIR,
    SIDE_SUFFIX,
    load_labeling_offsets,
    save_labeling_offset,
)

_DEFAULT_FPS = 30.0
_GRID_DT = 1.0 / 30.0  # 상호상관용 균일 시간축 간격(초). 최종 결과는 정수 프레임으로 반올림된다.
_DEFAULT_MAX_LAG_SEC = 2.0
# label_frames.py의 QSpinBox 범위와 동일 — 그 스핀박스에 그대로 넣을 수 있는 값인지 확인용.
_OFFSET_MIN = -30
_OFFSET_MAX = 30
_LOW_CONFIDENCE_SCORE = 0.3


@dataclass(frozen=True)
class OffsetEstimate:
    """한 base의 카메라 지연 추정 결과."""

    base: str
    offset_frames: int  # label_frames.py "지연 오프셋(프레임)"과 동일 정의 (g = side_map[k] - offset)
    offset_frames_unclamped: int
    delay_seconds: float  # 측면이 메인보다 지연된 시간(초). 음수면 측면이 오히려 앞선 것.
    score: float  # 최적 시차에서의 정규화 상관계수(-1~1)
    ticks_per_sec: float  # 전역 frame_index 기준 초당 틱 수 (frames.csv로부터 실측)
    n_samples: int  # 상호상관에 쓰인 공통 시간축 샘플 수
    low_confidence: bool
    out_of_range: bool
    # 디버그 플롯용 원시 데이터. --plot을 안 쓰면 None으로 남겨 메모리를 아낀다.
    main_grid: np.ndarray | None = None
    side_grid: np.ndarray | None = None
    scores: np.ndarray | None = None
    max_lag_samples: int | None = None


# ----------------------------------------------------------------------
# 프레임 정렬 (reconstruct_stroke.py의 side_ok 필터링과 같은 원리, main_ok에도 적용)
# ----------------------------------------------------------------------

def _load_track_timestamps(frames_csv: Path, ok_column: str) -> list[tuple[int, float]] | None:
    """``{base}_frames.csv``에서 ``ok_column==1``인 행만 순서대로 골라 (frame_index, elapsed_sec)
    쌍의 목록을 만든다.

    ``record_dual.py``는 프레임 루프 한 틱마다 frames.csv에 한 행을 쓰고, 그 틱에 실제로
    해당 스트림 프레임을 받았을 때만 영상에도 한 프레임을 쓴다. 따라서 영상의 k번째
    (0-index) 프레임에 대응하는 전역 frame_index·시각은, frames.csv에서 그 스트림의
    ok 플래그가 1인 행을 순서대로 센 k번째 행이다.
    """
    if not frames_csv.exists():
        return None
    rows: list[tuple[int, float]] = []
    with open(frames_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if int(row.get(ok_column, 1)):
                    rows.append((int(row["frame_index"]), float(row["elapsed_sec"])))
            except (KeyError, ValueError, TypeError):
                continue
    return rows or None


def _ticks_per_sec(frames_csv: Path, fallback_fps: float) -> float:
    """frames.csv 전체 구간으로부터 전역 frame_index의 초당 증가율(틱/초)을 구한다.

    없거나 구간이 0에 가까우면 영상 자체의 fps로 근사한다.
    """
    if not frames_csv.exists():
        return fallback_fps
    indices: list[int] = []
    seconds: list[float] = []
    with open(frames_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                indices.append(int(row["frame_index"]))
                seconds.append(float(row["elapsed_sec"]))
            except (KeyError, ValueError, TypeError):
                continue
    if len(indices) < 2:
        return fallback_fps
    span_sec = max(seconds) - min(seconds)
    span_idx = max(indices) - min(indices)
    if span_sec < 1e-6:
        return fallback_fps
    return span_idx / span_sec


# ----------------------------------------------------------------------
# 움직임 신호 추출
# ----------------------------------------------------------------------

def _motion_signal(
    video_path: Path,
    track: list[tuple[int, float]] | None,
    *,
    min_detection_confidence: float,
    min_tracking_confidence: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """영상을 훑어 (시각[초], 검지 끝 프레임간 변위) 시계열을 만든다.

    변위는 MediaPipe 정규화 좌표(0~1, 해상도 무관) 기준 유클리드 거리다 — 메인·측면 두
    카메라의 해상도·화각이 달라도 "움직임의 시간적 패턴"만 비교하면 되므로 절대 픽셀
    단위를 맞출 필요가 없다. 손 미검출이거나 직전 프레임에 손이 없었으면 변위 0으로 둔다
    (튀는 값 대신 "움직임 없음"으로 취급 — 상호상관에서 잡음보다는 무해한 근사).

    반환: ``(시각 배열, 변위 배열, 총 프레임 수, 검출 프레임 수)``.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = _DEFAULT_FPS

    tracker = HandTracker(
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    times: list[float] = []
    motion: list[float] = []
    prev_xy: np.ndarray | None = None
    total = 0
    detected = 0
    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if track is not None and idx < len(track):
                _, t = track[idx]
            else:
                t = idx / fps

            hands = tracker.process(frame)
            if hands:
                xy = np.asarray(hands[0].normalized_landmarks[INDEX_TIP][:2], dtype=np.float64)
                m = float(np.linalg.norm(xy - prev_xy)) if prev_xy is not None else 0.0
                prev_xy = xy
                detected += 1
            else:
                m = 0.0
                prev_xy = None

            times.append(t)
            motion.append(m)
            total += 1
            idx += 1
    finally:
        cap.release()
        tracker.close()

    return np.asarray(times, dtype=np.float64), np.asarray(motion, dtype=np.float64), total, detected


def _resample_uniform(t: np.ndarray, x: np.ndarray, dt: float, lo: float, hi: float) -> np.ndarray:
    """불규칙 시각 ``t``의 신호 ``x``를 ``[lo, hi]`` 구간의 균일 시간축(간격 ``dt``)에 선형보간한다."""
    if hi <= lo:
        return np.zeros(0, dtype=np.float64)
    grid = np.arange(lo, hi, dt)
    return np.interp(grid, t, x)


# ----------------------------------------------------------------------
# 상호상관
# ----------------------------------------------------------------------

def _shifted_segments(a: np.ndarray, b: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray] | None:
    """``b[t] = a[t - lag]`` 가설(=b가 a보다 lag 샘플만큼 지연)을 검증할 두 구간을 잘라낸다."""
    if lag >= 0:
        m = min(len(a), len(b) - lag)
        if m <= 0:
            return None
        return a[:m], b[lag:lag + m]
    k = -lag
    m = min(len(b), len(a) - k)
    if m <= 0:
        return None
    return a[k:k + m], b[:m]


def _bounded_cross_correlation(
    a: np.ndarray, b: np.ndarray, max_lag_samples: int
) -> tuple[int, np.ndarray]:
    """``-max_lag_samples..max_lag_samples`` 범위에서 ``b``가 ``a``보다 몇 샘플 지연됐을 때
    가장 잘 겹치는지(정규화 상관계수 최대) 찾는다.

    반환: ``(최적 lag, lag별 정규화 상관계수 배열)``. lag > 0이면 b(측면)가 지연된 것이다.
    """
    a = a - a.mean()
    b = b - b.mean()
    lags = list(range(-max_lag_samples, max_lag_samples + 1))
    scores = np.full(len(lags), -np.inf, dtype=np.float64)
    for i, lag in enumerate(lags):
        segs = _shifted_segments(a, b, lag)
        if segs is None:
            continue
        seg_a, seg_b = segs
        if len(seg_a) < 2:
            continue
        na, nb = np.linalg.norm(seg_a), np.linalg.norm(seg_b)
        if na < 1e-9 or nb < 1e-9:
            scores[i] = 0.0
            continue
        scores[i] = float(np.dot(seg_a, seg_b) / (na * nb))
    best_i = int(np.argmax(scores))
    return lags[best_i], scores


# ----------------------------------------------------------------------
# base 단위 추정
# ----------------------------------------------------------------------

def estimate_offset(
    base: str,
    folder: Path,
    *,
    max_lag_sec: float = _DEFAULT_MAX_LAG_SEC,
    min_detection_confidence: float = 0.7,
    min_tracking_confidence: float = 0.6,
    keep_debug: bool = False,
) -> OffsetEstimate:
    main_path = folder / f"{base}{MAIN_SUFFIX}"
    side_path = folder / f"{base}{SIDE_SUFFIX}"
    frames_path = folder / f"{base}{FRAMES_SUFFIX}"
    if not main_path.is_file() or not side_path.is_file():
        raise SystemExit(f"{base}: main/side 영상이 모두 있어야 합니다 ({main_path.name}, {side_path.name})")

    main_track = _load_track_timestamps(frames_path, "main_ok")
    side_track = _load_track_timestamps(frames_path, "side_ok")
    ticks_per_sec = _ticks_per_sec(frames_path, _DEFAULT_FPS)

    main_t, main_x, main_total, main_detected = _motion_signal(
        main_path, main_track,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    side_t, side_x, side_total, side_detected = _motion_signal(
        side_path, side_track,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    print(
        f"  [{base}] main {main_total}프레임(검출 {main_detected}) / "
        f"side {side_total}프레임(검출 {side_detected}) / ticks/sec={ticks_per_sec:.2f}"
    )

    lo = max(main_t.min(), side_t.min())
    hi = min(main_t.max(), side_t.max())
    main_grid = _resample_uniform(main_t, main_x, _GRID_DT, lo, hi)
    side_grid = _resample_uniform(side_t, side_x, _GRID_DT, lo, hi)
    n = min(len(main_grid), len(side_grid))
    main_grid, side_grid = main_grid[:n], side_grid[:n]
    if n < 10:
        raise SystemExit(f"{base}: 두 영상의 겹치는 구간이 너무 짧습니다 ({n} 샘플) — frames.csv를 확인하세요.")

    max_lag_samples = max(1, round(max_lag_sec / _GRID_DT))
    best_lag, scores = _bounded_cross_correlation(main_grid, side_grid, max_lag_samples)
    score = float(scores[best_lag + max_lag_samples])
    delay_seconds = best_lag * _GRID_DT

    offset_unclamped = int(round(delay_seconds * ticks_per_sec))
    offset = max(_OFFSET_MIN, min(_OFFSET_MAX, offset_unclamped))

    return OffsetEstimate(
        base=base,
        offset_frames=offset,
        offset_frames_unclamped=offset_unclamped,
        delay_seconds=delay_seconds,
        score=score,
        ticks_per_sec=ticks_per_sec,
        n_samples=n,
        low_confidence=score < _LOW_CONFIDENCE_SCORE,
        out_of_range=offset_unclamped != offset,
        main_grid=main_grid if keep_debug else None,
        side_grid=side_grid if keep_debug else None,
        scores=scores if keep_debug else None,
        max_lag_samples=max_lag_samples if keep_debug else None,
    )


def _plot(folder: Path, result: OffsetEstimate) -> Path:
    """정렬 전/후 신호와 lag별 상관계수를 그린 PNG를 저장한다 (영문 라벨만 — matplotlib
    기본 폰트에 한글 글리프가 없어 두부(모지박스)가 뜬다, 코드베이스 다른 오버레이와 동일 규칙)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    main_grid, side_grid = result.main_grid, result.side_grid
    scores, max_lag_samples = result.scores, result.max_lag_samples
    assert main_grid is not None and side_grid is not None and scores is not None
    n = len(main_grid)
    shift = round(result.delay_seconds / _GRID_DT)
    aligned_side = np.roll(side_grid, -shift) if shift else side_grid

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6))
    t = np.arange(n) * _GRID_DT
    ax1.plot(t, main_grid, label="main motion", color="tab:blue", linewidth=1)
    ax1.plot(t, side_grid, label="side motion (raw)", color="tab:orange", linewidth=1, alpha=0.5)
    ax1.plot(t, aligned_side, label="side motion (shifted)", color="tab:green", linewidth=1)
    ax1.set_title(f"{result.base}: motion signals (delay={result.delay_seconds:+.3f}s, score={result.score:.2f})")
    ax1.set_xlabel("time (s)")
    ax1.legend(loc="upper right", fontsize=8)

    lags_sec = np.arange(-max_lag_samples, max_lag_samples + 1) * _GRID_DT
    ax2.plot(lags_sec, scores, color="tab:purple", linewidth=1)
    ax2.axvline(result.delay_seconds, color="tab:red", linestyle="--", linewidth=1, label="best lag")
    ax2.set_title("normalized cross-correlation vs lag")
    ax2.set_xlabel("lag (s, positive = side delayed)")
    ax2.set_ylabel("score")
    ax2.legend(loc="upper right", fontsize=8)

    out_path = folder / f"{result.base}_offset_debug.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _find_bases(paths: list[str], folder: Path) -> list[str]:
    if not paths:
        return sorted(p.name[: -len(MAIN_SUFFIX)] for p in folder.glob(f"*{MAIN_SUFFIX}"))
    return list(paths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "bases", nargs="*",
        help="처리할 base명(들). 미지정 시 --folder 아래 모든 *_main.mp4 일괄 처리",
    )
    parser.add_argument("--folder", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--max-lag-sec", type=float, default=_DEFAULT_MAX_LAG_SEC)
    parser.add_argument("--min-detection-confidence", type=float, default=0.7)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.6)
    parser.add_argument(
        "--save", action="store_true",
        help=f"결과를 {OFFSETS_FILENAME}에 기록 (label_frames.py가 다음에 열 때 자동 복원)",
    )
    parser.add_argument("--plot", action="store_true", help="정렬 신호·상관계수 그래프 PNG 저장")
    args = parser.parse_args()

    folder = Path(args.folder)
    bases = _find_bases(args.bases, folder)
    if not bases:
        print(f"처리할 *{MAIN_SUFFIX} 파일이 없습니다: {folder}")
        return 1

    offsets_path = folder / OFFSETS_FILENAME
    existing = load_labeling_offsets(offsets_path)

    for base in bases:
        result = estimate_offset(
            base, folder,
            max_lag_sec=args.max_lag_sec,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            keep_debug=args.plot,
        )
        flags = []
        if result.low_confidence:
            flags.append(f"주의: 상관계수 낮음({result.score:.2f} < {_LOW_CONFIDENCE_SCORE}) — 육안 검산 권장")
        if result.out_of_range:
            flags.append(
                f"주의: 추정값 {result.offset_frames_unclamped}프레임이 스핀박스 범위"
                f"({_OFFSET_MIN}~{_OFFSET_MAX})를 벗어나 {result.offset_frames}로 클램프됨"
            )
        prev = existing.get(base)
        prev_note = f", 기존 저장값={prev}" if prev is not None else ""
        print(
            f"[{base}] 지연 {result.delay_seconds:+.3f}s -> offset={result.offset_frames}프레임 "
            f"(score={result.score:.2f}, n={result.n_samples}){prev_note}"
        )
        for flag in flags:
            print(f"  {flag}")

        if args.save:
            save_labeling_offset(offsets_path, base, result.offset_frames)
            print(f"  저장 완료: {offsets_path} [{base}] = {result.offset_frames}")

        if args.plot:
            plot_path = _plot(folder, result)
            print(f"  플롯 저장: {plot_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
