"""3D 접촉 판정 실측 검증 — 합성으로는 답할 수 없는 "실제로 쓸 만한가"를 잰다.

``core/contact3d.py``의 **수학**은 합성 ground truth로 오차 0.00mm까지 검증됐다. 하지만
실제 정확도를 좌우하는 것은 MediaPipe world landmarks가 프레임마다 얼마나 흔들리는가이며,
그건 실제 카메라로만 잴 수 있다. 이 스크립트가 그걸 측정한다.

측정 절차 (각 단계 기본 4초, 손을 **가만히** 유지할 것)
    1) 접촉: 캘리브레이션한 영역 안에서 손끝을 책상에 대고 유지
    2) 비접촉: 같은 위치에서 손끝을 약 3cm 띄운 채 유지

산출 지표
    - 접촉/비접촉 각각의 높이 중앙값과 표준편차(σ) — σ가 프레임 간 노이즈다
    - **분리도 d' = |중앙값 차이| / 평균 σ** — 클수록 두 상태를 잘 가른다
    - 같은 구간에서 계산한 pen_ratio의 d'와 비교 → 3D가 실제로 더 나은지 판단
    - 접촉 상태 높이의 편향(0에서 얼마나 벗어나는지) — 영점 보정으로 흡수될 양

판정 기준: d' >= 3 이면 히스테리시스로 안정적인 판정이 가능한 수준, 1.5 미만이면
그 신호로는 접촉 판정이 어렵다.

실행:
    python -m tools.validate_contact3d
    python -m tools.validate_contact3d --camera 1 --seconds 5
"""

from __future__ import annotations

import argparse
import statistics
import time

import cv2
import numpy as np

from core.camera import DEFAULT_INTRINSICS_PATH, CameraIntrinsics
from core.contact3d import (
    DEFAULT_PLANE_SIZE_MM,
    Contact3DError,
    Contact3DEstimator,
    estimate_plane_pose,
)
from core.geometry import DEFAULT_CALIBRATION_PATH, PerspectiveCalibration
from core.pen_state import compute_pen_ratio
from core.tracker import HandTracker

PHASES = (
    ("contact", "손끝을 책상(캘리브레이션 영역 안)에 **대고** 가만히 유지하세요"),
    ("hover", "같은 위치에서 손끝을 약 3cm **띄운 채** 가만히 유지하세요"),
)


def _d_prime(a: list[float], b: list[float]) -> float:
    """두 표본 분포의 분리도 = |중앙값 차| / 평균 표준편차."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = (statistics.pstdev(a) + statistics.pstdev(b)) / 2
    if pooled < 1e-9:
        return float("inf")
    return abs(statistics.median(a) - statistics.median(b)) / pooled


def _verdict(d: float) -> str:
    if not np.isfinite(d):
        return "판정 불가"
    if d >= 3.0:
        return "양호 (안정적 판정 가능)"
    if d >= 1.5:
        return "경계 (히스테리시스 폭을 넓게 잡아야 함)"
    return "불충분 (이 신호로는 접촉 판정 어려움)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--seconds", type=float, default=4.0, help="단계별 수집 시간(초)")
    parser.add_argument("--calibration", type=str, default=str(DEFAULT_CALIBRATION_PATH))
    parser.add_argument("--intrinsics", type=str, default=str(DEFAULT_INTRINSICS_PATH))
    parser.add_argument(
        "--plane-size", type=float, nargs=2, metavar=("W_MM", "H_MM"),
        default=list(DEFAULT_PLANE_SIZE_MM),
    )
    args = parser.parse_args()

    try:
        calibration = PerspectiveCalibration.load(args.calibration)
    except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
        print(f"캘리브레이션을 읽을 수 없습니다 ({args.calibration}): {exc}")
        print("먼저 `python -m tools.calibrate` 또는 `python -m controller.main`의 K 키로 4점을 지정하세요.")
        return 1

    plane_size = tuple(calibration.plane_size_mm or args.plane_size)

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        print(f"카메라 {args.camera}을(를) 열 수 없습니다.")
        return 1
    ok, frame = camera.read()
    if not ok:
        print("카메라 프레임을 읽지 못했습니다.")
        camera.release()
        return 1
    height, width = frame.shape[:2]

    intrinsics = CameraIntrinsics.load_or_estimate(width, height, path=args.intrinsics)
    if intrinsics.approximate:
        print("[주의] 정식 카메라 캘리브레이션이 없어 화각 60° 가정으로 근사합니다 — "
              "높이의 절대값은 비례 오차를 가질 수 있습니다(영점 보정으로 대부분 흡수).")

    try:
        plane_pose = estimate_plane_pose(
            calibration.src_points.as_array(), intrinsics, size_mm=plane_size
        )
    except Contact3DError as exc:
        print(f"평면 자세 추정 실패: {exc}")
        camera.release()
        return 1

    print(f"평면: {plane_size[0]:.0f}x{plane_size[1]:.0f}mm, "
          f"재투영오차 {plane_pose.reprojection_error_px:.2f}px, 해상도 {width}x{height}")

    estimator = Contact3DEstimator(plane_pose, intrinsics)
    tracker = HandTracker(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6)

    window = "Validate 3D contact - follow the prompt, Q to abort"
    results: dict[str, dict] = {}

    try:
        for phase, prompt in PHASES:
            heights: list[float] = []
            ratios: list[float] = []
            reproj: list[float] = []
            missing = 0
            # 준비 시간 + 수집 시간
            start = time.perf_counter()
            collecting_from = start + 2.0
            deadline = collecting_from + args.seconds
            while True:
                ok, frame = camera.read()
                if not ok:
                    break
                if args.mirror:
                    frame = cv2.flip(frame, 1)
                now = time.perf_counter()
                hands = tracker.process(frame)
                collecting = now >= collecting_from

                if hands:
                    hand = hands[0]
                    ratios_val = compute_pen_ratio(hand.normalized_landmarks, width, height)
                    sample = estimator.measure(hand)
                    if collecting:
                        ratios.append(ratios_val)
                        if sample is None:
                            missing += 1
                        else:
                            heights.append(sample.raw_height_mm)
                            reproj.append(sample.reprojection_error_px)
                    tip = hand.index_fingertip
                    cv2.circle(frame, tip, 10, (255, 180, 0), 2)
                    if sample is not None:
                        cv2.putText(frame, f"H {sample.raw_height_mm:+7.1f}mm", (15, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 210, 210), 2, cv2.LINE_AA)
                elif collecting:
                    missing += 1

                remain = deadline - now
                banner = prompt if collecting else f"{prompt}  (준비 {collecting_from - now:.1f}s)"
                cv2.putText(frame, f"[{phase}] {banner}", (15, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                if collecting:
                    cv2.putText(frame, f"수집 중... {max(remain, 0):.1f}s  (n={len(heights)})", (15, 75),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 60), 2, cv2.LINE_AA)
                cv2.imshow(window, frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("중단됨.")
                    return 1
                if now >= deadline:
                    break

            results[phase] = {
                "heights": heights, "ratios": ratios, "reproj": reproj, "missing": missing,
            }
            print(f"[{phase}] 수집 완료: 높이 표본 {len(heights)}개, "
                  f"3D 실패/손실 프레임 {missing}개")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        tracker.close()

    contact, hover = results.get("contact"), results.get("hover")
    if not contact or not hover or len(contact["heights"]) < 10 or len(hover["heights"]) < 10:
        print("\n표본이 부족해 분석할 수 없습니다 — 손이 안정적으로 검출되는 구도에서 다시 시도하세요.")
        return 1

    ch, hh = contact["heights"], hover["heights"]
    cr, hr = contact["ratios"], hover["ratios"]

    print("\n" + "=" * 74)
    print("3D 높이 신호 (mm)")
    print("=" * 74)
    print(f"  {'단계':<8}{'중앙값':>10}{'표준편차σ':>12}{'최소':>10}{'최대':>10}{'n':>7}")
    for label, s in (("접촉", ch), ("비접촉", hh)):
        print(f"  {label:<8}{statistics.median(s):>10.1f}{statistics.pstdev(s):>12.1f}"
              f"{min(s):>10.1f}{max(s):>10.1f}{len(s):>7}")
    sep = statistics.median(hh) - statistics.median(ch)
    d3 = _d_prime(ch, hh)
    print(f"\n  분리(비접촉−접촉) = {sep:+.1f} mm")
    print(f"  분리도 d' = {d3:.2f}  →  {_verdict(d3)}")
    print(f"  접촉 시 측정 높이 편향 = {statistics.median(ch):+.1f} mm "
          f"(영점 보정이 이 값을 0으로 맞춘다)")
    if reproj_all := (contact["reproj"] + hover["reproj"]):
        print(f"  손 자세 재투영 오차 평균 = {statistics.mean(reproj_all):.1f} px")

    print("\n" + "=" * 74)
    print("비교: 기존 pen_ratio 신호")
    print("=" * 74)
    dr = _d_prime(cr, hr)
    print(f"  {'단계':<8}{'중앙값':>10}{'표준편차σ':>12}")
    for label, s in (("접촉", cr), ("비접촉", hr)):
        print(f"  {label:<8}{statistics.median(s):>10.3f}{statistics.pstdev(s):>12.3f}")
    print(f"\n  분리도 d' = {dr:.2f}  →  {_verdict(dr)}")

    print("\n" + "=" * 74)
    print("결론")
    print("=" * 74)
    if np.isfinite(d3) and np.isfinite(dr):
        if d3 > dr * 1.2:
            print(f"  3D 높이(d'={d3:.2f})가 pen_ratio(d'={dr:.2f})보다 접촉을 잘 구분합니다 — 3D 사용 권장.")
        elif dr > d3 * 1.2:
            print(f"  pen_ratio(d'={dr:.2f})가 3D 높이(d'={d3:.2f})보다 낫습니다 — "
                  "카메라 각도/거리를 바꾸거나 정식 카메라 캘리브레이션 후 재측정해 보세요.")
        else:
            print(f"  두 신호의 분리도가 비슷합니다 (3D={d3:.2f}, pen_ratio={dr:.2f}).")
    n_missing = contact["missing"] + hover["missing"]
    if n_missing:
        total = n_missing + len(ch) + len(hh)
        print(f"  3D 측정 실패 프레임 {n_missing}/{total} — 이 프레임들은 pen_ratio로 자동 폴백됩니다.")
    if np.isfinite(d3) and d3 >= 1.5:
        margin = 0.18 * abs(sep)
        mid = (statistics.median(ch) + statistics.median(hh)) / 2 - statistics.median(ch)
        print(f"  권장 접촉 문턱(영점 보정 기준): down≈{mid - margin:.0f}mm, up≈{mid + margin:.0f}mm")
        print("  → `controller.main`에서 T 키를 누르면 이 값이 자동으로 계산·적용됩니다.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
