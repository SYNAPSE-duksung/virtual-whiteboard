"""Main control loop: webcam -> hand tracking -> stroke rendering.

``WhiteboardSession``(``controller/session.py``)이 트래킹·펜 판정(AUTO/MANUAL)·캔버스·
4점 캘리브레이션 평면 게이팅을 관리한다. 아키텍처·모드 설명·키 조작표·실행 옵션은
``controller/README.md`` 참고.

실행: ``python -m controller.main`` (옵션: ``--camera``, ``--mirror``, ``--calibration``)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from controller.session import WhiteboardSession
from core.camera import DEFAULT_INTRINSICS_PATH
from core.contact3d import DEFAULT_PLANE_SIZE_MM
from core.geometry import DEFAULT_CALIBRATION_PATH

_WINDOW_NAME = "Virtual Whiteboard - M mode / K calib / T touch-calib / SPACE pen / O ocr / C clear / S save / Q quit"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="camera device index")
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="flip the frame horizontally (selfie view; off by default for the desk view)",
    )
    parser.add_argument(
        "--flip-vertical",
        action="store_true",
        help="flip the frame vertically (oblique webcam upside-down fix; independent of --mirror)",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default=str(DEFAULT_CALIBRATION_PATH),
        help="캘리브레이션 저장/로드 경로 (기본 output/calibration.json)",
    )
    parser.add_argument(
        "--plane-size",
        type=float,
        nargs=2,
        metavar=("WIDTH_MM", "HEIGHT_MM"),
        default=list(DEFAULT_PLANE_SIZE_MM),
        help="캘리브레이션할 사각형의 실제 물리 크기(mm). 기본 297 210 = A4 가로. "
             "3D 접촉 판정의 정확도가 이 값에 비례하므로 실제로 찍는 영역과 맞출 것",
    )
    parser.add_argument(
        "--intrinsics",
        type=str,
        default=str(DEFAULT_INTRINSICS_PATH),
        help="카메라 내부 파라미터 JSON 경로. 없으면 화각 가정으로 근사한다",
    )
    parser.add_argument(
        "--no-3d",
        action="store_true",
        help="3D 접촉 판정을 끄고 기존 pen_ratio 휴리스틱만 사용",
    )
    parser.add_argument(
        "--debug-3d",
        action="store_true",
        help="3D 디버그 오버레이(평면 수선 벡터·높이 눈금·수치)를 켠 채로 시작 (실행 중 D로 토글)",
    )
    args = parser.parse_args()

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        print(f"카메라 {args.camera}을(를) 열 수 없습니다.")
        return 1

    print("M: 모드 전환 | K: 4점 캘리브레이션 | T: 접촉 캘리브레이션(3D 영점 포함) | D: 3D 디버그 표시 | O: OCR 트리거 | C: 지우기 | S: 저장 | Q: 종료")
    if not args.no_3d:
        print(f"[3D] 평면 실제 크기 {args.plane_size[0]:.0f}x{args.plane_size[1]:.0f}mm 가정 "
              f"— 다르면 --plane-size 로 지정하세요 (A4 가로면 그대로 두면 됩니다)")
    previous_time = time.perf_counter()
    last_touch_message: tuple[str, bool] | None = None
    last_pipeline_status: str | None = None
    try:
        with WhiteboardSession(
            calibration_path=args.calibration,
            intrinsics_path=args.intrinsics,
            plane_size_mm=(args.plane_size[0], args.plane_size[1]),
            use_3d_contact=not args.no_3d,
        ) as session:
            if args.debug_3d:
                session.set_debug_3d(True)
            cv2.namedWindow(_WINDOW_NAME)
            cv2.setMouseCallback(
                _WINDOW_NAME,
                lambda event, x, y, flags, userdata: session.add_calibration_point(x, y)
                if event == cv2.EVENT_LBUTTONDOWN
                else None,
            )

            if session.needs_calibration:
                # 세션 시작 시 유효한 캘리브레이션이 없으면(최초 실행) 1회 강제한다.
                # 이후에는 K로 언제든 재캘리브레이션할 수 있다.
                print("[캘리브레이션] 저장된 캘리브레이션이 없어 최초 지정을 시작합니다.")
                print("  TL→TR→BR→BL 순서로 클릭 (Z 취소 / X 리셋 / Enter,S 적용 / Esc는 무시 — 최초 1회는 필수)")
                session.start_calibration()

            while True:
                ok, frame = camera.read()
                if not ok:
                    print("카메라 프레임을 읽지 못했습니다.")
                    return 1
                if args.mirror:
                    frame = cv2.flip(frame, 1)
                if args.flip_vertical:
                    frame = cv2.flip(frame, 0)

                annotated = session.process_frame(frame)

                # 터치 캘리브레이션은 키 입력이 아니라 시간 경과로 자동 완료되므로,
                # 결과 메시지가 바뀔 때만 콘솔에도 한 번 출력한다 (화면 표시는 항상 됨).
                touch_msg = session.touch_message
                if touch_msg is not None and touch_msg != last_touch_message:
                    print(f"[터치 캘리브레이션] {touch_msg[0]}")
                last_touch_message = touch_msg

                # 상태가 바뀔 때만 출력 (매 프레임 폴링이라 스팸 방지).
                debug = session.debug
                if debug.pipeline_status != last_pipeline_status:
                    if debug.pipeline_status == "DONE":
                        suffix = f": {debug.corrected_text}"
                    elif debug.pipeline_status == "ERROR":
                        suffix = f": {debug.pipeline_error}"
                    else:
                        suffix = ""
                    print(f"[OCR/LLM] {debug.pipeline_status}{suffix}")
                last_pipeline_status = debug.pipeline_status

                current_time = time.perf_counter()
                fps = 1.0 / max(current_time - previous_time, 1e-6)
                previous_time = current_time
                cv2.putText(annotated, f"FPS {fps:.1f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow(_WINDOW_NAME, annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if session.is_calibrating:
                    if key == ord("z"):
                        session.undo_calibration_point()
                    elif key == ord("x"):
                        session.reset_calibration_points()
                    elif key == ord(" "):  # SPACE: 현재 손끝 위치에 점 찍기 (마우스 없이)
                        if not session.add_calibration_point_at_fingertip():
                            print("[캘리브레이션] 손이 검출되지 않아 점을 찍을 수 없습니다 — 클릭으로 지정하세요.")
                    elif key in (13, ord("s")):  # Enter 또는 S: 적용
                        if session.confirm_calibration():
                            cal = session.calibration
                            print(f"[캘리브레이션] 적용 완료 (dst={cal.dst_size if cal else '?'})")
                        else:
                            msg = session.calibration_message
                            print(f"[캘리브레이션] {msg[0] if msg else '4점을 모두 지정하세요.'}")
                    elif key == 27:  # Esc: 취소
                        if session.has_calibration:
                            session.cancel_calibration()
                            print("[캘리브레이션] 취소 — 기존 캘리브레이션 유지")
                        else:
                            print("[캘리브레이션] 아직 캘리브레이션이 없어 취소할 수 없습니다 — 4점을 지정하세요.")
                elif session.is_touch_calibrating:
                    if key == 27:  # Esc: 취소
                        session.cancel_touch_calibration()
                        print("[터치 캘리브레이션] 취소 — 기존 임계값 유지")
                elif key == ord("k"):
                    session.start_calibration()
                    print("[캘리브레이션] 시작 — 클릭 또는 손끝+SPACE로 TL→TR→BR→BL 순서 지정 (Z 취소 / X 리셋 / Enter,S 적용 / Esc 취소)")
                elif key == ord("t"):
                    session.start_touch_calibration()
                    print(f"[터치 캘리브레이션] 시작 — 먼저 {session.touch_sample_seconds:.1f}초간 손가락을 화면 위에 띄우세요 (Esc 취소)")
                elif key == ord("d"):
                    on = session.toggle_debug_3d()
                    print(f"[3D 디버그] {'표시' if on else '숨김'}"
                          + ("" if session.has_3d_contact else " (※ 3D 접촉 판정이 비활성 상태입니다)"))
                elif key == ord("m"):
                    auto = session.toggle_mode()
                    print(f"모드 전환: {'자동(손끝 판정)' if auto else '수동(SPACE 펜)'}")
                elif key == ord("o"):
                    if not session.trigger_ocr():
                        print("[OCR/LLM] 이전 작업 처리 중 — 잠시 후 다시 시도하세요.")
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
