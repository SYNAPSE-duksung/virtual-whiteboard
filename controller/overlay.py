"""``WhiteboardSession``이 쓰는 cv2 프레임 오버레이 그리기 함수 모음.

캘리브레이션/터치캘리브레이션 안내, 3D 디버그 패널(수선 벡터·높이 눈금·수치)을
프레임 픽셀에 직접 그린다. 순수 렌더링 전용 — 세션 상태를 직접 읽지 않고 필요한
값을 매개변수로 받는다(``controller/session.py``에서 호출).
"""

from __future__ import annotations

import cv2
import numpy as np

from core.contact3d import MAX_REPROJECTION_ERROR_PX
from core.geometry import CalibrationPicker
from core.pen_tracker import PenTracker

__all__ = [
    "PLANE_OUTLINE_COLOR",
    "draw_3d_debug",
    "draw_height_gauge",
    "draw_calibration_overlay",
    "draw_touch_calibration_overlay",
]

CAL_POINT_COLOR = (0, 215, 255)
CAL_LINE_COLOR = (60, 220, 60)
CAL_TEXT_COLOR = (255, 255, 255)
CAL_OK_COLOR = (0, 200, 0)
CAL_ERR_COLOR = (0, 0, 255)
PLANE_OUTLINE_COLOR = (255, 120, 0)
CAL_CURSOR_COLOR = (255, 180, 0)
TOUCH_BAR_BG = (70, 70, 70)
TOUCH_BAR_FG = (0, 210, 210)
# 3D 디버그 오버레이 색 (BGR)
DEBUG_NORMAL_COLOR = (255, 200, 0)  # 수선 벡터 (비접촉)
DEBUG_CONTACT_COLOR = (0, 220, 60)  # 수선 벡터 (접촉) / down 문턱
DEBUG_UP_COLOR = (0, 165, 255)  # up 문턱
DEBUG_RULER_COLOR = (200, 200, 200)  # 높이 눈금
# 높이 게이지 범위(mm). 접촉 문턱(기본 12/22mm)과 손을 살짝 든 정도(~60mm)가 모두 들어오고,
# 영점이 덜 맞았을 때의 음수 높이도 보이도록 아래로 조금 여유를 둔다.
GAUGE_MIN_MM, GAUGE_MAX_MM = -20.0, 80.0
GAUGE_TICKS_MM = (-20.0, 0.0, 20.0, 40.0, 60.0, 80.0)


def _px(point) -> tuple[int, int]:
    """부동소수 이미지 좌표를 OpenCV 그리기용 정수 픽셀로."""
    return (int(round(float(point[0]))), int(round(float(point[1]))))


def _wrap_text(text: str, *, font_scale: float, thickness: int, max_width: int) -> list[str]:
    """``max_width`` px 안에 들어가도록 단어 단위로 줄바꿈 (실제 렌더 폭을 잰다).

    프레임 폭이 좁을 때(캘리브레이션/3D 디버그 안내 문구는 길다) 텍스트가 프레임
    바깥으로 잘려 나가는 것을 막는다. 폭 계산에 ``cv2.getTextSize``를 써서 문자
    수가 아니라 실제 렌더 폭 기준으로 자른다.
    """
    words = text.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        (w, _), _ = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        if w > max_width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _put_text_wrapped(
    annotated: np.ndarray, text: str, origin: tuple[int, int], *,
    font_scale: float, color: tuple[int, int, int], thickness: int, line_height: int,
    max_width: int | None = None,
) -> int:
    """``_wrap_text``로 줄바꿈해 여러 줄로 그리고, 그린 줄 수를 반환한다."""
    x, y = origin
    if max_width is None:
        max_width = annotated.shape[1] - x - 15  # 오른쪽 여백 15px
    lines = _wrap_text(text, font_scale=font_scale, thickness=thickness, max_width=max_width)
    for i, line in enumerate(lines):
        cv2.putText(annotated, line, (x, y + line_height * i),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    return len(lines)


def draw_3d_debug(
    annotated: np.ndarray, result, *, tracker: PenTracker, plane_size_mm: tuple[float, float],
    show_panel: bool = True,
) -> None:
    """손끝→평면 수선(직교 벡터)과 높이 눈금(그래픽)을 겹쳐 그린다 (제자리 수정).

    3D가 꺼져 있거나 이번 프레임에서 자세 추정이 실패했으면 그 사실 자체를 표시한다 —
    디버그를 켰는데 아무것도 안 보이면 "왜 안 보이는지"를 알 수 없기 때문이다.

    ``show_panel=False``면 수치 텍스트 패널(우상단 8줄 요약, off/실패 사유 문구)은
    프레임에 굽지 않는다 — PyQt 앱처럼 같은 정보를 Qt 라벨로 별도 표시할 수 있는
    호출부는 영상 위에 큰 텍스트 박스가 겹치지 않도록 끌 수 있다. 손끝-평면 수선·
    눈금·높이 게이지 같은 그래픽 오버레이는 손끝 위치와 겹쳐 봐야 의미가 있어
    ``show_panel`` 값과 무관하게 계속 그린다.
    """
    height, width = annotated.shape[:2]
    debug = getattr(result, "contact_debug", None)
    if debug is None:
        if not show_panel:
            return
        # OpenCV의 Hershey 폰트에는 한글 글리프가 없어 ????로 깨진다 — 오버레이 문구는
        # 전부 ASCII로 쓴다(코드베이스의 다른 오버레이도 같은 규칙).
        estimator = tracker.contact_estimator
        if not tracker.has_3d_contact:
            reason = "3D DEBUG: OFF - need 4-pt calibration + plane size + intrinsics (press K)"
        elif estimator is not None and estimator.last_reprojection_error_px is not None:
            # 문턱(기본 25px)을 얼마나 넘겼는지 보여준다 — 근소 초과면 카메라 캘리브레이션
            # 부족, 크게 초과하면 손 자세 자체가 안 잡힌 것(가림·빠른 움직임 등)일 가능성이 높다.
            reason = (
                f"3D DEBUG: hand reproj {estimator.last_reprojection_error_px:.0f}px "
                f"> {MAX_REPROJECTION_ERROR_PX:.0f}px threshold (falling back to pen_ratio)"
            )
        else:
            reason = "3D DEBUG: hand pose solve failed this frame (falling back to pen_ratio)"
        _put_text_wrapped(annotated, reason, (15, 30), font_scale=0.55,
                           color=CAL_ERR_COLOR, thickness=2, line_height=26)
        return

    tip = np.array(debug.tip_px, dtype=np.float64)
    foot = np.array(debug.foot_px, dtype=np.float64)
    contact = result.pen_down
    axis_color = DEBUG_CONTACT_COLOR if contact else DEBUG_NORMAL_COLOR

    # 수선(평면에 직교하는 벡터): 수선의 발 → 손끝. 이게 곧 "평면에서 얼마나 떴는가"다.
    cv2.line(annotated, _px(foot), _px(tip), axis_color, 2, cv2.LINE_AA)
    # 수선의 발 = 손끝을 평면에 정사영한 위치. 평면 위에 있음을 십자로 표시.
    fx, fy = _px(foot)
    cv2.line(annotated, (fx - 9, fy), (fx + 9, fy), axis_color, 1, cv2.LINE_AA)
    cv2.line(annotated, (fx, fy - 9), (fx, fy + 9), axis_color, 1, cv2.LINE_AA)
    cv2.circle(annotated, _px(tip), 5, axis_color, -1, cv2.LINE_AA)

    # 눈금을 그릴 방향: 수선의 화면상 방향에 수직한 단위벡터.
    axis = tip - foot
    norm = float(np.linalg.norm(axis))
    perp = np.array([-axis[1], axis[0]]) / norm if norm > 1e-6 else np.array([1.0, 0.0])

    for mm, px in debug.ruler_px:
        p = np.array(px, dtype=np.float64)
        cv2.line(annotated, _px(p - perp * 6), _px(p + perp * 6),
                 DEBUG_RULER_COLOR, 1, cv2.LINE_AA)
        cv2.putText(annotated, f"{mm:.0f}", _px(p + perp * 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, DEBUG_RULER_COLOR, 1, cv2.LINE_AA)

    # 접촉 문턱은 눈금보다 굵고 길게 — 손끝이 이 선 아래로 내려가면 pen down이다.
    for label, mm, px in debug.threshold_px:
        p = np.array(px, dtype=np.float64)
        color = DEBUG_CONTACT_COLOR if label == "down" else DEBUG_UP_COLOR
        cv2.line(annotated, _px(p - perp * 14), _px(p + perp * 14), color, 2, cv2.LINE_AA)
        cv2.putText(annotated, f"{label} {mm:.0f}", _px(p - perp * 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    estimator = tracker.contact_estimator
    detector = tracker.contact_detector

    if show_panel:
        plane_x, plane_y = debug.plane_xy_mm
        plane_w, plane_h = plane_size_mm
        zero = "n/a  (press T)" if estimator is None else f"{estimator.zero_offset_mm:+8.1f} mm"
        lines = [
            ("3D DEBUG   (D to hide)", CAL_TEXT_COLOR),
            (f"height     {debug.height_mm:+8.1f} mm", axis_color),
            (f"raw height {debug.raw_height_mm:+8.1f} mm", CAL_TEXT_COLOR),
            (f"zero offs  {zero}",
             CAL_TEXT_COLOR if estimator is not None else CAL_ERR_COLOR),
            (f"thresh d/u {detector.down_mm:.1f} / {detector.up_mm:.1f} mm", DEBUG_RULER_COLOR),
            (f"plane XY   {plane_x:6.0f},{plane_y:6.0f} of {plane_w:.0f}x{plane_h:.0f}"
             + ("" if debug.inside_plane else "  OUT"),
             CAL_TEXT_COLOR if debug.inside_plane else CAL_ERR_COLOR),
            (f"hand reproj{debug.reprojection_error_px:7.1f} px", CAL_TEXT_COLOR),
            (f"source {result.contact_source}   contact {'YES' if contact else 'no'}", axis_color),
        ]
        # 좌하단 상태 문구와 겹치지 않도록 우상단에 패널을 띄운다.
        # panel_w는 고정값 대신 실제 렌더 폭 중 최댓값으로 맞춰, 값이 길어져도(음수 좌표·
        # 큰 숫자 등) 배경 박스 밖으로 글씨가 삐져나오지 않게 한다.
        line_h = 23
        text_widths = [
            cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.47, 1)[0][0] for text, _ in lines
        ]
        panel_w = max(text_widths, default=200) + 10
        x0 = max(width - panel_w - 12, 15)
        y0 = 18
        overlay = annotated.copy()
        cv2.rectangle(overlay, (x0 - 10, y0 - 6),
                      (x0 + panel_w + 2, y0 + line_h * len(lines)), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0, annotated)
        for i, (text, color) in enumerate(lines):
            cv2.putText(annotated, text, (x0, y0 + line_h * (i + 1) - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.47, color, 1, cv2.LINE_AA)

    draw_height_gauge(annotated, debug.height_mm, detector, contact)


def draw_height_gauge(annotated: np.ndarray, height_mm: float, detector, contact: bool) -> None:
    """화면 왼쪽에 높이 막대 게이지를 그린다 (투영과 무관한 읽기 전용 눈금).

    카메라가 책상을 거의 수직으로 내려다보면 평면의 법선이 시선과 거의 나란해져,
    화면에 투영된 수선이 몇 픽셀로 찌부러져 눈으로 읽을 수 없다. 이 게이지는 3D
    기하를 투영하지 않고 높이 값만 막대로 그리므로 그 구도에서도 항상 읽힌다.
    """
    frame_h = annotated.shape[0]
    x, top, bottom = 40, 120, frame_h - 140
    span = bottom - top
    lo, hi = GAUGE_MIN_MM, GAUGE_MAX_MM

    def y_of(mm: float) -> int:
        # 위쪽이 높은 높이(hi), 아래쪽이 lo. 범위를 벗어나면 끝에 붙인다.
        frac = (float(mm) - lo) / (hi - lo)
        return int(round(bottom - max(0.0, min(1.0, frac)) * span))

    overlay = annotated.copy()
    cv2.rectangle(overlay, (x - 16, top - 22), (x + 78, bottom + 20), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)
    cv2.line(annotated, (x, top), (x, bottom), DEBUG_RULER_COLOR, 1, cv2.LINE_AA)
    cv2.putText(annotated, "mm", (x - 12, top - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, DEBUG_RULER_COLOR, 1, cv2.LINE_AA)

    for mm in GAUGE_TICKS_MM:
        y = y_of(mm)
        cv2.line(annotated, (x - 5, y), (x + 5, y), DEBUG_RULER_COLOR, 1, cv2.LINE_AA)
        cv2.putText(annotated, f"{mm:+.0f}", (x + 10, y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, DEBUG_RULER_COLOR, 1, cv2.LINE_AA)

    # 접촉 문턱: 이 두 선 사이가 히스테리시스 구간이다.
    for mm, color in ((detector.down_mm, DEBUG_CONTACT_COLOR),
                      (detector.up_mm, DEBUG_UP_COLOR)):
        y = y_of(mm)
        cv2.line(annotated, (x - 12, y), (x + 12, y), color, 2, cv2.LINE_AA)

    # 현재 높이 표시자.
    y = y_of(height_mm)
    color = DEBUG_CONTACT_COLOR if contact else DEBUG_NORMAL_COLOR
    cv2.line(annotated, (x - 14, y), (x + 14, y), color, 2, cv2.LINE_AA)
    cv2.circle(annotated, (x, y), 5, color, -1, cv2.LINE_AA)
    cv2.putText(annotated, f"{height_mm:+.1f}", (x - 14, bottom + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_calibration_overlay(
    frame: np.ndarray,
    *,
    picker: CalibrationPicker | None,
    fingertip: tuple[int, int] | None,
    message: tuple[str, bool] | None,
) -> np.ndarray:
    """캘리브레이션 모드의 점·연결선·손끝 커서·안내 문구·메시지를 그려 반환한다."""
    annotated = frame.copy()
    if picker is None:
        return annotated

    points = picker.points
    for i, (x, y) in enumerate(points):
        pt = (int(round(x)), int(round(y)))
        cv2.circle(annotated, pt, 8, CAL_POINT_COLOR, -1)
        cv2.putText(
            annotated, str(i + 1), (pt[0] + 10, pt[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, CAL_POINT_COLOR, 2, cv2.LINE_AA,
        )
    if len(points) >= 2:
        pts_arr = np.array([[int(x), int(y)] for x, y in points], dtype=np.int32)
        cv2.polylines(annotated, [pts_arr], picker.is_complete, CAL_LINE_COLOR, 2, cv2.LINE_AA)

    # 손끝 커서: 마우스 없이도 손가락으로 점을 찍을 수 있다는 것을 보여준다.
    if fingertip is not None:
        cv2.circle(annotated, fingertip, 12, CAL_CURSOR_COLOR, 2)
        cv2.circle(annotated, fingertip, 2, CAL_CURSOR_COLOR, -1)

    # 이 오버레이는 cv2.putText로 그려지므로 ASCII로만 작성한다 (Hershey 폰트에
    # 한글 글리프가 없어 한글을 넣으면 ?????로 깨진다). 콘솔 print()는 한글 그대로 둔다.
    if not picker.is_complete:
        guide = f"[CALIBRATION] next: {picker.next_label} ({picker.count}/4) - click or SPACE at fingertip"
    else:
        guide = "[CALIBRATION] 4 points done - Enter/S apply / Z undo / X reset / Esc cancel"
    # main()의 FPS 카운터가 (15,30)을 이미 쓰므로 그 아래에 그린다.
    guide_lines = _put_text_wrapped(annotated, guide, (15, 60), font_scale=0.65,
                                     color=CAL_TEXT_COLOR, thickness=2, line_height=28)
    if fingertip is None:
        _put_text_wrapped(
            annotated, "(no hand detected - SPACE unavailable, click still works)",
            (15, 60 + 28 * guide_lines), font_scale=0.55,
            color=(150, 150, 255), thickness=1, line_height=24,
        )

    if message:
        text, ok = message
        color = CAL_OK_COLOR if ok else CAL_ERR_COLOR
        lines = _wrap_text(text, font_scale=0.6, thickness=2,
                            max_width=annotated.shape[1] - 30)
        base_y = annotated.shape[0] - 20 - 26 * (len(lines) - 1)
        for i, line in enumerate(lines):
            cv2.putText(annotated, line, (15, base_y + 26 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return annotated


def draw_touch_calibration_overlay(
    frame: np.ndarray,
    *,
    phase: str | None,
    progress: float,
    fingertip: tuple[int, int] | None,
    message: tuple[str, bool] | None,
) -> np.ndarray:
    """터치 캘리브레이션 모드의 단계 안내·진행률 게이지·손끝 커서를 그려 반환한다."""
    annotated = frame.copy()
    if fingertip is not None:
        cv2.circle(annotated, fingertip, 12, CAL_CURSOR_COLOR, 2)

    # ASCII로만 작성 (Hershey 폰트에 한글 글리프 없음 — 위 캘리브레이션 오버레이와 동일 규칙).
    if phase == "hover":
        title = "[TOUCH CAL 1/2] hold finger above the surface, still (not touching)"
    elif phase == "touch":
        title = "[TOUCH CAL 2/2] press finger on the surface and hold still"
    else:
        title = "[TOUCH CALIBRATION]"
    cv2.putText(annotated, title, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, CAL_TEXT_COLOR, 2, cv2.LINE_AA)
    cv2.putText(
        annotated, "Esc cancel", (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, CAL_TEXT_COLOR, 1, cv2.LINE_AA,
    )

    # 진행률 게이지 (0~1)
    bar_x, bar_y, bar_w, bar_h = 15, 100, 300, 14
    cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), TOUCH_BAR_BG, -1)
    fill_w = int(bar_w * progress)
    if fill_w > 0:
        cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), TOUCH_BAR_FG, -1)
    cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), CAL_TEXT_COLOR, 1)

    if message:
        text, ok = message
        color = CAL_OK_COLOR if ok else CAL_ERR_COLOR
        # 긴 결과 문구는 화면 폭에 맞춰 두 줄로 접는다.
        mid = len(text) // 2
        split = text.rfind(" ", 0, mid + 20) if len(text) > 55 else -1
        lines = [text] if split == -1 else [text[:split], text[split + 1:]]
        base_y = annotated.shape[0] - 20 - 22 * (len(lines) - 1)
        for i, line in enumerate(lines):
            cv2.putText(
                annotated, line, (15, base_y + 22 * i),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
            )
    return annotated
