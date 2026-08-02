"""4점 캘리브레이션 지정 UI (4주차 A파트).

카메라로 책상 위 필기 영역의 네 모서리를 클릭해 지정하면, ``core.geometry``로 투시 변환을
계산해 ``output/calibration.json``에 저장한다. 저장된 파일은
``core.geometry.PerspectiveCalibration.load()``로 다른 스크립트(예: 향후 pen 판정 강화,
결과 렌더링)에서 재사용한다.

점을 찍는 순서는 항상 **TL → TR → BR → BL** (좌상단부터 시계방향)이다.

조작:
    마우스 좌클릭  : 점 추가 (최대 4개)
    z             : 마지막 점 취소
    r             : 전체 리셋
    p             : 정류(bird's-eye) 미리보기 창 토글 (4점 완료 후)
    s             : 저장 (``output/calibration.json``, 4점 완료 + 유효할 때만)
    q             : 종료

실행:
    python -m tools.calibrate
    python -m tools.calibrate --camera 1 --output output/calibration.json
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from core.geometry import DEFAULT_CALIBRATION_PATH, CalibrationPicker

_POINT_COLOR = (0, 215, 255)
_LINE_COLOR = (60, 220, 60)
_TEXT_COLOR = (255, 255, 255)
_ERROR_COLOR = (0, 0, 255)
_OK_COLOR = (0, 200, 0)


def draw_overlay(
    frame: np.ndarray,
    picker: CalibrationPicker,
    *,
    message: str | None = None,
    message_ok: bool = True,
) -> np.ndarray:
    """picker 상태(점·연결선·다음 안내)와 하단 메시지를 프레임에 그려 반환.

    cv2 드로잉 API만 쓰는 순수 함수라 디스플레이 없이도 테스트 가능하다
    (``cv2.imshow``/``waitKey``는 여기서 호출하지 않는다).
    """
    annotated = frame.copy()
    points = picker.points

    for i, (x, y) in enumerate(points):
        pt = (int(round(x)), int(round(y)))
        cv2.circle(annotated, pt, 8, _POINT_COLOR, -1)
        cv2.putText(
            annotated, str(i + 1), (pt[0] + 10, pt[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, _POINT_COLOR, 2, cv2.LINE_AA,
        )

    if len(points) >= 2:
        closed = picker.is_complete  # 4점 완료 시에만 사각형을 닫는다
        pts_arr = np.array([[int(x), int(y)] for x, y in points], dtype=np.int32)
        cv2.polylines(annotated, [pts_arr], closed, _LINE_COLOR, 2, cv2.LINE_AA)

    # ASCII로만 작성 (Hershey 폰트에 한글 글리프가 없어 한글은 ?????로 깨진다).
    guide = (
        f"next click: {picker.next_label} ({picker.count}/4)"
        if not picker.is_complete
        else "4 points done - s save / p preview / z undo / r reset"
    )
    cv2.putText(annotated, guide, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, _TEXT_COLOR, 2, cv2.LINE_AA)

    if message:
        color = _OK_COLOR if message_ok else _ERROR_COLOR
        cv2.putText(
            annotated, message, (15, annotated.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA,
        )
    return annotated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="카메라 인덱스 (기본 0)")
    parser.add_argument(
        "--mirror", action="store_true",
        help="좌우 반전(셀피 구도). 기본값은 반전 없음 — 책상을 내려다보는 구도 전제",
    )
    parser.add_argument("--output", type=str, default=str(DEFAULT_CALIBRATION_PATH))
    args = parser.parse_args()

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        print(f"카메라 {args.camera}을(를) 열 수 없습니다.")
        return 1

    picker = CalibrationPicker()
    message: str | None = None
    message_ok = True
    show_preview = False

    def on_mouse(event, x, y, flags, userdata) -> None:  # noqa: ANN001 (cv2 콜백 시그니처)
        nonlocal message, message_ok
        if event == cv2.EVENT_LBUTTONDOWN:
            picker.add_point(x, y)
            message, message_ok = None, True

    window = "Calibrate - click TL/TR/BR/BL, s save, p preview, z undo, r reset, q quit"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    print("좌클릭으로 TL→TR→BR→BL 순서로 4점을 지정하세요. (z 취소, r 리셋, p 미리보기, s 저장, q 종료)")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("카메라 프레임을 읽지 못했습니다.")
                return 1
            if args.mirror:
                frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            annotated = draw_overlay(frame, picker, message=message, message_ok=message_ok)
            cv2.imshow(window, annotated)

            if show_preview and picker.is_complete:
                cal, err = picker.try_build(source_frame_size=(width, height))
                if cal is not None:
                    cv2.imshow("Rectified preview", cal.warp_frame(frame))
                elif err:
                    message, message_ok = err, False
                    show_preview = False
                    cv2.destroyWindow("Rectified preview")

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("z"):
                picker.undo()
                message = None
                if show_preview:
                    show_preview = False
                    cv2.destroyWindow("Rectified preview")
            elif key == ord("r"):
                picker.reset()
                message = None
                if show_preview:
                    show_preview = False
                    cv2.destroyWindow("Rectified preview")
            elif key == ord("p"):
                if not picker.is_complete:
                    # 화면 오버레이에 표시되므로 ASCII로만 작성 (한글 글리프 없음).
                    message, message_ok = "Place all 4 points first.", False
                else:
                    show_preview = not show_preview
                    if not show_preview:
                        cv2.destroyWindow("Rectified preview")
            elif key == ord("s"):
                cal, err = picker.try_build(source_frame_size=(width, height))
                if cal is None:
                    message, message_ok = err or "Place all 4 points first.", False
                else:
                    path = cal.save(args.output)
                    message, message_ok = f"Saved: {path} (dst={cal.dst_size})", True
                    print(f"[캘리브레이션] 저장: {path}")
                    print(f"  src_points(TL,TR,BR,BL)={cal.src_points.as_array().tolist()}")
                    print(f"  dst_size={cal.dst_size}")
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
