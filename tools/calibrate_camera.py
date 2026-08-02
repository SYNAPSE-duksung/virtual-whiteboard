"""체스보드 카메라 캘리브레이션 — 정식 intrinsics를 구해 ``output/camera_intrinsics.json``에 저장.

3D 접촉 판정(``core.contact3d``)은 카메라 초점거리·주점을 알아야 한다. 없으면 화각 60°
가정으로 근사하는데, 초점거리가 20% 틀리면 복원 높이도 비례해 치우친다(영점 보정으로
대부분 흡수되지만, 남는 비례 오차까지 없애려면 이 도구로 제대로 재는 게 좋다).

준비물: 체스보드 패턴을 A4에 인쇄해 **평평한 판에 붙인다**(구겨지면 정확도가 떨어진다).
기본값은 내부 코너 9x6 (= 10x7 칸) 패턴이다.

사용법
    python -m tools.calibrate_camera
    - 체스보드를 카메라에 비추면 코너가 검출되어 초록색으로 표시된다.
    - SPACE: 현재 프레임을 캡처 (각도·거리·화면 위치를 **다양하게** 바꿔가며 15장 이상 권장)
    - C: 캡처한 장면들로 캘리브레이션 실행 후 저장
    - Z: 마지막 캡처 취소 / Q: 종료

    다른 패턴이면 `--cols`, `--rows`(내부 코너 개수), `--square-mm`(칸 한 변 mm)를 지정한다.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from core.camera import DEFAULT_INTRINSICS_PATH, CameraIntrinsics

MIN_CAPTURES = 8
RECOMMENDED_CAPTURES = 15


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--cols", type=int, default=9, help="체스보드 내부 코너 수(가로)")
    parser.add_argument("--rows", type=int, default=6, help="체스보드 내부 코너 수(세로)")
    parser.add_argument("--square-mm", type=float, default=25.0, help="칸 한 변 길이(mm)")
    parser.add_argument("--output", type=str, default=str(DEFAULT_INTRINSICS_PATH))
    args = parser.parse_args()

    pattern = (args.cols, args.rows)
    # 체스보드 로컬 좌표(z=0 평면). 실제 칸 크기를 곱해 mm 단위로 만든다.
    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square_mm

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        print(f"카메라 {args.camera}을(를) 열 수 없습니다.")
        return 1

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    size: tuple[int, int] | None = None
    message = ""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    window = "Camera calibration - SPACE capture / C calibrate / Z undo / Q quit"

    print(f"체스보드 내부 코너 {args.cols}x{args.rows}, 칸 {args.square_mm}mm")
    print(f"SPACE로 {RECOMMENDED_CAPTURES}장 이상 캡처한 뒤 C를 누르세요 (각도·거리를 다양하게).")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("카메라 프레임을 읽지 못했습니다.")
                return 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            size = gray.shape[::-1]
            found, corners = cv2.findChessboardCorners(
                gray, pattern,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK,
            )
            display = frame.copy()
            if found:
                refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                cv2.drawChessboardCorners(display, pattern, refined, found)

            cv2.putText(display, f"캡처 {len(img_points)}장 (권장 {RECOMMENDED_CAPTURES}장 이상)",
                        (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 220, 60) if found else (200, 200, 200), 2, cv2.LINE_AA)
            cv2.putText(display, "체스보드 검출됨 — SPACE로 캡처" if found else "체스보드가 보이지 않습니다",
                        (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 220, 60) if found else (0, 0, 255), 2, cv2.LINE_AA)
            if message:
                cv2.putText(display, message, (15, display.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 220), 2, cv2.LINE_AA)
            cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                if not found:
                    message = "체스보드가 검출되지 않아 캡처할 수 없습니다."
                else:
                    obj_points.append(objp.copy())
                    img_points.append(refined)
                    message = f"캡처 {len(img_points)}장"
                    print(f"  캡처 {len(img_points)}")
            elif key == ord("z"):
                if obj_points:
                    obj_points.pop(); img_points.pop()
                    message = f"마지막 캡처 취소 (남은 {len(img_points)}장)"
            elif key == ord("c"):
                if len(img_points) < MIN_CAPTURES:
                    message = f"최소 {MIN_CAPTURES}장이 필요합니다 (현재 {len(img_points)}장)"
                    continue
                print("캘리브레이션 계산 중...")
                rms, mtx, dist, _, _ = cv2.calibrateCamera(
                    obj_points, img_points, size, None, None
                )
                intr = CameraIntrinsics(
                    fx=float(mtx[0, 0]), fy=float(mtx[1, 1]),
                    cx=float(mtx[0, 2]), cy=float(mtx[1, 2]),
                    width=size[0], height=size[1],
                    dist_coeffs=tuple(float(v) for v in dist.ravel()[:5]),
                    approximate=False,
                )
                path = intr.save(args.output)
                approx = CameraIntrinsics.from_fov(size[0], size[1])
                print(f"\n저장: {path}")
                print(f"  RMS 재투영 오차 = {rms:.3f} px  (0.5 이하면 양호)")
                print(f"  fx={intr.fx:.1f} fy={intr.fy:.1f} cx={intr.cx:.1f} cy={intr.cy:.1f}")
                print(f"  화각 60° 근사값 대비 초점거리 차이: {100 * (intr.fx / approx.fx - 1):+.1f}%")
                print("  → 이제 controller.main / tools.validate_contact3d가 이 값을 자동으로 사용합니다.")
                message = f"저장 완료 (RMS {rms:.2f}px)"
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
