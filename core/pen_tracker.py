"""고수준 펜 추적기.

``HandTracker``(랜드마크) + ``FingertipSmoother``(스무딩) + ``PenStateDetector``(자동
pen-up/down)를 하나로 조합한다. controller/ui는 이 모듈만 사용하면 되고, 프레임을 넣으면
스무딩된 손끝 좌표·펜 상태·지우기 제스처가 담긴 ``PenFrame``을 돌려받는다.

``tracker.py``의 규약(트래킹 손실 시 예외 대신 빈 결과)을 따른다 — 손을 잃으면
``PenFrame.empty()``를 반환하고 내부 필터/상태를 리셋한다.

판정 신호는 세 가지이며, 우선순위대로 시도하다 안 되면 다음으로 폴백한다
-----------------------------------------------------------------
1. **측면(폰) 카메라 판정(최우선)**: ``side_frame``이 주어지고 측면 기준선
   (``core.side_contact.SideBaseline``)이 설정돼 있으면, 그 프레임에서 손끝-책상 거리(px)로
   접촉을 판정한다. 듀얼 카메라 계획(CLAUDE.md 참고)의 핵심 신호 — 측면 구도는 손끝-책상
   간격이 원근 왜곡 없이 보여, 노트북 카메라의 3D 판정이 가장 불안정한 자세(손이 카메라
   쪽으로 접히는 자세)에서도 안정적이다. 그 프레임에 측면 손이 검출되지 않으면 아래로 폴백.
2. **3D 접촉 판정**: 캘리브레이션에 평면의 실제 물리 크기(``plane_size_mm``)가 있고
   카메라 intrinsics를 알 때, ``core.contact3d``가 MediaPipe world landmarks로 손끝의
   **평면 위 실제 높이(mm)** 를 복원해 그 값으로 접촉을 판정한다. 진짜 3차원 판정이다.
3. **pen_ratio 판정(최종 폴백)**: 위 조건이 안 되거나 자세 추정이 실패한 프레임에서는
   기존 손가락 굽힘 비율 휴리스틱을 쓴다.

어느 쪽이 쓰였는지는 ``PenFrame.contact_source``(``"side"``|``"3d"``|``"ratio"``)로 확인할 수 있다.

**평면 범위 게이팅**은 두 경우 모두에 적용된다 — 손끝이 캘리브레이션한 사각형 밖이면
판정과 무관하게 ``pen_down``을 False로 강제한다.

⚠️ 3D 판정의 정확도: 복원 높이는 MediaPipe world landmarks(단일 RGB 추정)의 손 모델
정확도와 intrinsics 가정에 비례해 치우친다 — 손 모델 스케일이 10% 틀리면 높이가 40mm
넘게 틀어진다. 그래서 ``Contact3DEstimator``의 **영점 보정**(표면을 실제로 짚었을 때의
측정값을 0으로 맞춤)이 필수이며, 보정 후에는 그 계통 오차가 대부분 상쇄된다. 남는 것은
프레임 간 랜덤 노이즈이고, 그 크기는 ``tools/validate_contact3d.py``로 실측해야 한다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2

from core.camera import CameraIntrinsics
from core.contact3d import (
    Contact3DError,
    Contact3DEstimator,
    ContactDebugGeometry,
    ContactDetector,
    estimate_plane_pose,
)
from core.filters import FingertipSmoother
from core.geometry import PerspectiveCalibration
from core.pen_state import INDEX_TIP, PenStateDetector, compute_pen_ratio, is_open_hand
from core.side_contact import SideBaseline, SideContactDetector
from core.side_contact import measure as measure_side
from core.tracker import HandTracker


@dataclass(frozen=True, slots=True)
class PenFrame:
    """한 프레임의 펜 추적 결과."""

    hand_detected: bool
    fingertip: tuple[int, int] | None  # 스무딩된 픽셀 좌표 = 펜 입력점 (filtered_x/y)
    raw_fingertip: tuple[float, float] | None  # 필터 이전 원본 픽셀 좌표 (float)
    raw_z: float | None  # 검지 끝 정규화 z (깊이 근사, CSV 기록용)
    pen_down: bool  # 게이팅까지 반영된 최종 판정
    erase_gesture: bool
    pen_ratio: float | None
    in_bounds: bool | None  # None=캘리브레이션 미설정(게이팅 비활성). True/False=평면 내부 여부
    rectified: tuple[float, float] | None  # 손끝의 정류(캘리브레이션 평면) 좌표. 캘리브레이션 없으면 None
    #: 3D 복원 높이(mm, 영점 보정 후). 3D 판정이 불가한 프레임에서는 None.
    height_mm: float | None = None
    #: 영점 보정 전 원시 측정 높이(mm) — 영점 캘리브레이션 때 이 값을 모은다.
    raw_height_mm: float | None = None
    #: 이번 프레임 pen_down이 어느 신호로 결정됐는지: ``"side"`` | ``"3d"`` | ``"ratio"``.
    contact_source: str = "ratio"
    #: 3D 디버그 시각화용 기하 (``PenTracker.set_debug_3d(True)`` 일 때만 채워진다).
    contact_debug: ContactDebugGeometry | None = None
    #: 측면 카메라 기준 접촉 판정(히스테리시스 적용 후). 측면 신호를 못 얻은 프레임은 None.
    side_pen_down: bool | None = None
    #: 측면 카메라에서 손끝-기준선까지의 원시 거리(px). 측면 신호를 못 얻은 프레임은 None.
    side_distance_px: float | None = None

    @classmethod
    def empty(cls) -> "PenFrame":
        """트래킹 손실 프레임."""
        return cls(
            False, None, None, None, False, False, None, None, None, None, None, "ratio", None
        )


class PenTracker:
    """프레임 in → ``PenFrame`` out. 내부 상태(필터/히스테리시스)를 프레임 간 유지한다."""

    def __init__(
        self,
        *,
        max_num_hands: int = 1,
        min_cutoff: float = 1.0,
        beta: float = 0.3,
        d_cutoff: float = 1.0,
        pen_down_thresh: float = 0.55,
        pen_up_thresh: float = 0.70,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.6,
        calibration: PerspectiveCalibration | None = None,
        intrinsics: CameraIntrinsics | None = None,
        contact_down_mm: float | None = None,
        contact_up_mm: float | None = None,
        use_3d_contact: bool = True,
        side_baseline: SideBaseline | None = None,
        side_down_px: float | None = None,
        side_up_px: float | None = None,
    ) -> None:
        self._tracker = HandTracker(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._smoother = FingertipSmoother(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self._pen_state = PenStateDetector(down_thresh=pen_down_thresh, up_thresh=pen_up_thresh)
        self._use_3d_contact = use_3d_contact
        self._intrinsics = intrinsics
        contact_kwargs = {}
        if contact_down_mm is not None:
            contact_kwargs["down_mm"] = contact_down_mm
        if contact_up_mm is not None:
            contact_kwargs["up_mm"] = contact_up_mm
        self._contact_detector = ContactDetector(**contact_kwargs)
        self._contact_estimator: Contact3DEstimator | None = None
        # 디버그 시각화는 프레임마다 추가 투영 계산이 붙으므로 기본은 꺼 둔다.
        self._debug_3d = False
        self._calibration = None
        self._set_calibration_internal(calibration)

        # 측면(폰) 카메라 접촉 판정. 별도 물리 카메라이므로 MediaPipe 인스턴스도 메인과
        # 독립적으로 둬야 한다(같은 인스턴스를 두 스트림에 번갈아 먹이면 내부 트래킹 상태가
        # 섞인다) — 기준선이 설정될 때만 지연 생성한다(측면 카메라 미사용 시 비용 없음).
        self._side_min_detection_confidence = min_detection_confidence
        self._side_min_tracking_confidence = min_tracking_confidence
        self._side_tracker: HandTracker | None = None
        self._side_baseline: SideBaseline | None = None
        self._side_detector: SideContactDetector | None = None
        self.set_side_baseline(side_baseline, down_px=side_down_px, up_px=side_up_px)

    # ------------------------------------------------------------------
    # 캘리브레이션 / 3D 접촉 추정기
    # ------------------------------------------------------------------
    @property
    def calibration(self) -> PerspectiveCalibration | None:
        return self._calibration

    @property
    def intrinsics(self) -> CameraIntrinsics | None:
        return self._intrinsics

    @property
    def contact_estimator(self) -> Contact3DEstimator | None:
        """3D 접촉 추정기 (평면 크기·intrinsics가 갖춰졌을 때만 생성됨)."""
        return self._contact_estimator

    @property
    def has_3d_contact(self) -> bool:
        """이 트래커가 3D 접촉 판정을 쓸 수 있는 상태인지."""
        return self._contact_estimator is not None

    @property
    def contact_detector(self) -> ContactDetector:
        return self._contact_detector

    @property
    def debug_3d(self) -> bool:
        """3D 디버그 기하(``PenFrame.contact_debug``) 계산 여부."""
        return self._debug_3d

    def set_debug_3d(self, enabled: bool) -> None:
        self._debug_3d = bool(enabled)

    def _set_calibration_internal(self, calibration: PerspectiveCalibration | None) -> None:
        """캘리브레이션을 반영하고, 조건이 되면 3D 접촉 추정기를 (재)구성한다.

        3D 판정에는 ① 평면의 실제 물리 크기 ② 카메라 intrinsics 가 모두 필요하다.
        하나라도 없거나 평면 자세가 풀리지 않으면 추정기를 만들지 않고, ``process()``는
        기존 pen_ratio 판정으로 동작한다(조용한 폴백).
        """
        self._calibration = calibration
        self._contact_estimator = None
        if not self._use_3d_contact or calibration is None or self._intrinsics is None:
            return
        if calibration.plane_size_mm is None:
            return
        try:
            plane_pose = estimate_plane_pose(
                calibration.src_points.as_array(),
                self._intrinsics,
                size_mm=calibration.plane_size_mm,
            )
        except (Contact3DError, cv2.error):
            return
        self._contact_estimator = Contact3DEstimator(plane_pose, self._intrinsics)

    def set_calibration(self, calibration: PerspectiveCalibration | None) -> None:
        """캘리브레이션을 교체(또는 ``None``으로 게이팅 해제)한다.

        평면 경계가 바뀌면 진행 중이던 획/히스테리시스 상태가 새 경계 기준으로는
        무의미해지므로, 교체 시 필터·펜 상태를 함께 리셋한다. 3D 접촉 추정기도 새 평면으로
        다시 만들어지며, 이전 영점 보정값은 무효가 되므로 버려진다.
        """
        self._set_calibration_internal(calibration)
        self.reset()

    def set_intrinsics(self, intrinsics: CameraIntrinsics | None) -> None:
        """카메라 intrinsics를 교체하고 3D 접촉 추정기를 다시 구성한다."""
        self._intrinsics = intrinsics
        self._set_calibration_internal(self._calibration)
        self.reset()

    def set_contact_zero_offset(self, offset_mm: float) -> None:
        """3D 높이의 영점(표면을 짚었을 때의 측정값)을 설정한다. 추정기가 없으면 무시."""
        if self._contact_estimator is not None:
            self._contact_estimator.set_zero_offset(offset_mm)

    def set_contact_thresholds(self, *, down_mm: float, up_mm: float) -> None:
        """3D 접촉 히스테리시스 문턱(mm)을 교체한다."""
        self._contact_detector.set_thresholds(down_mm=down_mm, up_mm=up_mm)

    # ------------------------------------------------------------------
    # 측면(폰) 카메라 접촉 판정
    # ------------------------------------------------------------------
    @property
    def has_side_contact(self) -> bool:
        """측면 카메라 접촉 판정을 쓸 수 있는 상태인지(기준선이 설정됐는지)."""
        return self._side_baseline is not None

    @property
    def side_baseline(self) -> SideBaseline | None:
        return self._side_baseline

    @property
    def side_detector(self) -> SideContactDetector | None:
        return self._side_detector

    def set_side_baseline(
        self,
        baseline: SideBaseline | None,
        *,
        down_px: float | None = None,
        up_px: float | None = None,
    ) -> None:
        """측면 기준선을 설정(또는 ``None``으로 해제)한다.

        기준선을 처음 설정할 때만 측면 전용 ``HandTracker``를 만든다 — 측면 카메라를
        아예 안 쓰는 세션(대부분의 경우)에서 불필요한 MediaPipe 인스턴스 비용을 피한다.
        ``None``으로 해제하면 트래커도 함께 닫는다.
        """
        self._side_baseline = baseline
        if baseline is None:
            self._side_detector = None
            if self._side_tracker is not None:
                self._side_tracker.close()
                self._side_tracker = None
            return
        kwargs: dict[str, float] = {}
        if down_px is not None:
            kwargs["down_px"] = down_px
        if up_px is not None:
            kwargs["up_px"] = up_px
        self._side_detector = SideContactDetector(**kwargs)
        if self._side_tracker is None:
            self._side_tracker = HandTracker(
                max_num_hands=1,
                min_detection_confidence=self._side_min_detection_confidence,
                min_tracking_confidence=self._side_min_tracking_confidence,
            )

    def set_side_thresholds(self, *, down_px: float, up_px: float) -> None:
        """측면 접촉 히스테리시스 문턱(px)을 교체한다. 기준선이 없으면 무시."""
        if self._side_detector is not None:
            self._side_detector.set_thresholds(down_px=down_px, up_px=up_px)

    @property
    def down_thresh(self) -> float:
        return self._pen_state.down_thresh

    @property
    def up_thresh(self) -> float:
        return self._pen_state.up_thresh

    def set_thresholds(self, *, down_thresh: float, up_thresh: float) -> None:
        """pen_ratio 히스테리시스 임계값을 런타임에 교체한다 (예: 터치 캘리브레이션 결과 적용)."""
        self._pen_state.set_thresholds(down_thresh=down_thresh, up_thresh=up_thresh)

    def process(
        self, bgr_frame, *, side_frame=None, timestamp: float | None = None
    ) -> PenFrame:
        """BGR 프레임을 처리해 ``PenFrame`` 반환.

        ``side_frame``(측면 카메라 프레임)을 함께 주면, 기준선이 설정돼 있고 그 프레임에서
        측면 손이 검출되는 한 측면 신호가 3D/pen_ratio보다 우선한다. ``timestamp``는
        One Euro Filter의 시간 기준(초). 미지정 시 ``time.perf_counter()`` 사용.
        """
        hands = self._tracker.process(bgr_frame)
        if not hands:
            self._smoother.reset()
            self._pen_state.reset()
            self._contact_detector.reset()
            if self._side_detector is not None:
                self._side_detector.reset()
            return PenFrame.empty()

        hand = hands[0]
        height, width = bgr_frame.shape[:2]
        t = time.perf_counter() if timestamp is None else timestamp

        landmarks = hand.normalized_landmarks
        raw_x = float(landmarks[INDEX_TIP][0]) * width
        raw_y = float(landmarks[INDEX_TIP][1]) * height
        raw_z = float(landmarks[INDEX_TIP][2])
        smooth_x, smooth_y = self._smoother.update(t, raw_x, raw_y)
        fingertip = (int(round(smooth_x)), int(round(smooth_y)))

        erase_gesture = is_open_hand(landmarks)
        pen_ratio = compute_pen_ratio(landmarks, width, height)

        # 3D 접촉 측정 (가능한 프레임에서만). 실패하면 None → pen_ratio로 폴백한다.
        sample = None if self._contact_estimator is None else self._contact_estimator.measure(hand)
        height_mm = sample.height_mm if sample is not None else None
        raw_height_mm = sample.raw_height_mm if sample is not None else None

        contact_debug = None
        if self._debug_3d and sample is not None and self._contact_estimator is not None:
            # 판정에 실제로 쓰는 문턱을 그대로 넘겨, 화면 눈금과 판정이 어긋날 수 없게 한다.
            contact_debug = self._contact_estimator.debug_geometry(
                sample,
                thresholds=(
                    ("down", self._contact_detector.down_mm),
                    ("up", self._contact_detector.up_mm),
                ),
            )

        # 측면 카메라 측정 (기준선이 설정돼 있고 side_frame이 주어진 프레임에서만).
        # 그 프레임에 측면 손이 안 잡히면 이번 프레임만 3D/pen_ratio로 폴백한다.
        side_distance = None
        if self._side_tracker is not None and self._side_baseline is not None and side_frame is not None:
            side_hands = self._side_tracker.process(side_frame)
            if side_hands:
                side_distance = measure_side(side_hands[0], self._side_baseline)

        side_pen_down_signal: bool | None = None
        if erase_gesture:
            # 지우기 제스처 중에는 펜을 내리지 않는다.
            self._pen_state.reset()
            self._contact_detector.reset()
            if self._side_detector is not None:
                self._side_detector.reset()
            instant_pen_down = False
            contact_source = "side" if side_distance is not None else ("3d" if sample is not None else "ratio")
        elif side_distance is not None:
            instant_pen_down = self._side_detector.update(side_distance)
            side_pen_down_signal = instant_pen_down
            # 폴백 대비로 3D/pen_ratio 히스테리시스도 계속 최신화 (측면이 다음 프레임에
            # 실패해도 어긋난 상태에서 시작하지 않도록) — 3D 우선 로직과 같은 이유.
            if sample is not None:
                self._contact_detector.update(sample.height_mm)
            self._pen_state.update(pen_ratio)
            contact_source = "side"
        elif sample is not None:
            instant_pen_down = self._contact_detector.update(sample.height_mm)
            # 폴백 대비로 pen_ratio 히스테리시스도 계속 돌려 상태를 최신으로 유지한다
            # (다음 프레임에 3D가 실패해도 어긋난 상태에서 시작하지 않도록).
            self._pen_state.update(pen_ratio)
            contact_source = "3d"
        else:
            instant_pen_down = self._pen_state.update(pen_ratio)
            self._contact_detector.reset()
            contact_source = "ratio"

        in_bounds: bool | None = None
        rectified: tuple[float, float] | None = None
        if self._calibration is not None:
            in_bounds = self._calibration.contains((raw_x, raw_y))
            rx, ry = self._calibration.to_rectified((raw_x, raw_y))
            rectified = (float(rx), float(ry))

        pen_down = self._apply_bounds_gate(instant_pen_down, in_bounds)

        return PenFrame(
            hand_detected=True,
            fingertip=fingertip,
            raw_fingertip=(raw_x, raw_y),
            raw_z=raw_z,
            pen_down=pen_down,
            erase_gesture=erase_gesture,
            pen_ratio=pen_ratio,
            in_bounds=in_bounds,
            rectified=rectified,
            height_mm=height_mm,
            raw_height_mm=raw_height_mm,
            contact_source=contact_source,
            contact_debug=contact_debug,
            side_pen_down=side_pen_down_signal,
            side_distance_px=side_distance,
        )

    @staticmethod
    def _apply_bounds_gate(pen_down: bool, in_bounds: bool | None) -> bool:
        """캘리브레이션 평면 밖(``in_bounds is False``)이면 무조건 up. 미설정(None)이면 통과."""
        if in_bounds is False:
            return False
        return pen_down

    def reset(self) -> None:
        """필터·펜 상태를 초기화한다."""
        self._smoother.reset()
        self._pen_state.reset()
        self._contact_detector.reset()
        if self._side_detector is not None:
            self._side_detector.reset()

    def close(self) -> None:
        self._tracker.close()
        if self._side_tracker is not None:
            self._side_tracker.close()

    def __enter__(self) -> "PenTracker":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
