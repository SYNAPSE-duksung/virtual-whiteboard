"""진짜 3D 접촉 판정 — 손끝의 평면 위 **실제 높이(mm)** 를 복원한다.

기존 ``compute_pen_ratio``는 "손가락이 얼마나 굽었는가"라는 2D 대리 신호였다. 이 모듈은
그 대신 손끝이 책상 평면에서 **몇 mm 떠 있는지**를 직접 계산해, 그 값으로 pen up/down을
판정한다.

원리
----
1. **평면 자세**: 4점 캘리브레이션 코너(이미지 픽셀)와 그 사각형의 **실제 크기(mm)**,
   그리고 카메라 intrinsics로 ``cv2.solvePnP`` → 책상 평면의 3D 자세 ``(R_p, t_p)``.
2. **손 자세**: MediaPipe **world landmarks**(손 중심 원점, 미터 단위 metric 3D 손 모델)와
   같은 손의 이미지 픽셀 좌표 21쌍으로 다시 ``cv2.solvePnP`` → 손의 3D 자세 ``(R_h, t_h)``.
   world landmarks가 metric이라 스케일 모호성이 풀린다 — 이게 단일 RGB로 3D를 얻는 핵심.
3. **높이**: 손끝을 카메라 좌표로 옮기고(``R_h @ W_tip + t_h``), 다시 평면 좌표로 옮기면
   (``R_p^T @ (P_cam - t_p)``) z성분이 곧 평면 위 높이(mm)다. 부호는 카메라가 있는 쪽이
   양수가 되도록 정규화한다.

정확도와 영점 보정 (중요)
------------------------
MediaPipe의 world landmarks는 단일 RGB에서 **추정**한 3D라 절대 정확도에 한계가 있고,
근사 intrinsics를 쓰면 거리 전체가 비례해서 치우친다. 그래서 이 모듈은 "높이 0 = 접촉"을
그대로 믿지 않고, 사용자가 실제로 표면을 짚었을 때의 높이 측정값을 **영점(zero offset)** 으로
학습해 빼도록 설계했다(``set_zero_offset``). 즉 판정에 쓰는 값은

    유효 높이 = 측정 높이 − 영점

이고, 계통 오차(bias)는 영점이 흡수한다. 남는 것은 프레임 간 노이즈이며, 그 크기가 실제로
얼마인지는 ``tools/validate_contact3d.py``로 실측해야 한다 — 노이즈가 접촉 판정 문턱보다
크면 이 방식도 실패한다. 합성 데이터로는 수학만 검증할 수 있을 뿐이다.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.camera import CameraIntrinsics

# A4 용지 가로 배치 (297 x 210 mm). 책상에 A4를 놓고 그 네 모서리를 캘리브레이션하면
# 자를 대지 않고도 정확한 물리 크기를 쓸 수 있어 기본값으로 삼는다.
DEFAULT_PLANE_SIZE_MM: tuple[float, float] = (297.0, 210.0)

INDEX_TIP = 8

# 접촉 판정 기본 문턱(mm). 영점 보정 후의 유효 높이 기준이며, 히스테리시스로 채터링을 막는다.
DEFAULT_CONTACT_DOWN_MM = 12.0
DEFAULT_CONTACT_UP_MM = 22.0

# solvePnP 재투영 오차가 이보다 크면 자세 추정이 실패한 것으로 보고 버린다(픽셀).
MAX_REPROJECTION_ERROR_PX = 25.0

# 디버그 시각화에서 수선을 따라 찍는 유효 높이 눈금(mm). 접촉 문턱(12~22mm)이 가운데
# 오도록 잡아, 손끝이 문턱의 위/아래 어디에 있는지 한눈에 보이게 한다.
DEFAULT_RULER_MM: tuple[float, ...] = (0.0, 10.0, 20.0, 30.0, 50.0)


class Contact3DError(ValueError):
    """3D 자세 추정에 필요한 입력이 부족하거나 기하학적으로 풀리지 않을 때."""


@dataclass(frozen=True)
class Pose:
    """``X_cam = R @ X_obj + t`` 형태의 강체 변환."""

    rotation: np.ndarray  # (3, 3)
    translation: np.ndarray  # (3,) mm

    def to_camera(self, points: np.ndarray) -> np.ndarray:
        """물체 좌표 ``(N, 3)`` → 카메라 좌표 ``(N, 3)``."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        return pts @ self.rotation.T + self.translation

    def to_object(self, points: np.ndarray) -> np.ndarray:
        """카메라 좌표 ``(N, 3)`` → 물체 좌표 ``(N, 3)`` (역변환)."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        return (pts - self.translation) @ self.rotation


@dataclass(frozen=True)
class PlanePose:
    """책상 평면의 3D 자세 + 높이 부호 규약."""

    pose: Pose
    size_mm: tuple[float, float]
    #: ``+1``이면 평면 좌표 z가 그대로 "카메라 쪽 높이", ``-1``이면 부호를 뒤집어야 한다.
    height_sign: float
    reprojection_error_px: float

    def height_of(self, point_cam: np.ndarray) -> float:
        """카메라 좌표의 점 하나가 평면 위 몇 mm에 있는지 (카메라 쪽이 양수)."""
        local = self.pose.to_object(np.asarray(point_cam, dtype=np.float64).reshape(1, 3))
        return float(local[0, 2] * self.height_sign)

    def plane_xy_of(self, point_cam: np.ndarray) -> tuple[float, float]:
        """카메라 좌표의 점을 평면 위로 정사영한 (x, y) mm 좌표."""
        local = self.pose.to_object(np.asarray(point_cam, dtype=np.float64).reshape(1, 3))
        return float(local[0, 0]), float(local[0, 1])

    def point_at(self, plane_xy: tuple[float, float], height_mm: float) -> np.ndarray:
        """평면 좌표 ``(x, y)`` 위 높이 ``height_mm``인 점을 카메라 좌표로 (``height_of``의 역).

        디버그 시각화에서 "수선의 발"(높이 0)이나 문턱 높이의 위치를 화면에 찍는 데 쓴다.
        """
        local = np.array(
            [float(plane_xy[0]), float(plane_xy[1]), float(height_mm) * self.height_sign],
            dtype=np.float64,
        )
        return self.pose.to_camera(local.reshape(1, 3))[0]

    def contains_xy(self, plane_xy: tuple[float, float]) -> bool:
        """평면 좌표가 캘리브레이션한 사각형 내부인지."""
        x, y = plane_xy
        return 0.0 <= x <= self.size_mm[0] and 0.0 <= y <= self.size_mm[1]


def _reprojection_error(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> float:
    projected, _ = cv2.projectPoints(
        object_points, rvec, tvec, intrinsics.matrix, intrinsics.distortion
    )
    projected = projected.reshape(-1, 2)
    return float(np.mean(np.linalg.norm(projected - image_points.reshape(-1, 2), axis=1)))


def estimate_plane_pose(
    image_corners: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    size_mm: tuple[float, float] = DEFAULT_PLANE_SIZE_MM,
) -> PlanePose:
    """4점 캘리브레이션 코너(TL, TR, BR, BL)로부터 평면의 3D 자세를 구한다.

    ``size_mm``은 그 사각형의 실제 물리 크기 (가로, 세로) 밀리미터. 이 값이 틀리면 복원된
    거리·높이가 통째로 그 비율만큼 틀어진다.
    """
    corners = np.asarray(image_corners, dtype=np.float64).reshape(-1, 2)
    if corners.shape != (4, 2):
        raise Contact3DError(f"4개 코너가 필요합니다: shape={corners.shape}")
    width_mm, height_mm = float(size_mm[0]), float(size_mm[1])
    if width_mm <= 0 or height_mm <= 0:
        raise Contact3DError(f"평면 크기는 양수여야 합니다: {size_mm}")

    # 평면 로컬 좌표: TL이 원점, +x는 오른쪽(TR 방향), +y는 아래쪽(BL 방향), z=0이 평면.
    object_points = np.array(
        [[0.0, 0.0, 0.0], [width_mm, 0.0, 0.0], [width_mm, height_mm, 0.0], [0.0, height_mm, 0.0]],
        dtype=np.float64,
    )

    # 4개 공면 점은 자세가 두 갈래로 모호할 수 있고, 솔버마다 어느 쪽을 고르는지가 다르다
    # (특히 SOLVEPNP_IPPE는 잘못된 분기를 고르는 경우가 관측됨). 여러 솔버를 돌려
    # **재투영 오차가 가장 작은 해**를 채택해 그 모호성을 실측으로 끊는다.
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for flag in (cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_IPPE):
        try:
            ok, rvec, tvec = cv2.solvePnP(
                object_points, corners, intrinsics.matrix, intrinsics.distortion, flags=flag
            )
        except cv2.error:
            continue
        if not ok:
            continue
        error = _reprojection_error(object_points, corners, rvec, tvec, intrinsics)
        if best is None or error < best[0]:
            best = (error, rvec, tvec)

    if best is None:
        raise Contact3DError("평면 자세(solvePnP) 추정에 실패했습니다 — 코너 좌표를 확인하세요.")

    error, rvec, tvec = best
    if error > MAX_REPROJECTION_ERROR_PX:
        raise Contact3DError(
            f"평면 자세 재투영 오차가 너무 큽니다 ({error:.1f}px) — "
            "코너 순서(TL→TR→BR→BL)나 평면 실제 크기가 맞는지 확인하세요."
        )

    rotation, _ = cv2.Rodrigues(rvec)
    pose = Pose(rotation=rotation, translation=tvec.reshape(3))

    # 카메라 원점을 평면 좌표로 옮겨, 카메라가 평면의 어느 쪽에 있는지로 높이 부호를 정한다.
    camera_in_plane = pose.to_object(np.zeros((1, 3)))[0]
    height_sign = 1.0 if camera_in_plane[2] >= 0 else -1.0

    return PlanePose(
        pose=pose, size_mm=(width_mm, height_mm),
        height_sign=height_sign, reprojection_error_px=error,
    )


def estimate_hand_pose(
    world_landmarks_mm: np.ndarray,
    pixel_landmarks: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> tuple[Pose, float]:
    """metric world landmarks ↔ 이미지 좌표 21쌍으로 손의 3D 자세를 구한다.

    반환값은 ``(자세, 재투영오차px)``. world landmarks가 미터가 아니라 **밀리미터**로
    들어와야 한다(``TrackedHand.world_landmarks_mm``).
    """
    obj = np.asarray(world_landmarks_mm, dtype=np.float64).reshape(-1, 3)
    img = np.asarray(pixel_landmarks, dtype=np.float64).reshape(-1, 2)
    if obj.shape[0] < 6 or obj.shape[0] != img.shape[0]:
        raise Contact3DError(
            f"랜드마크 대응쌍이 부족합니다: world={obj.shape}, image={img.shape}"
        )

    ok, rvec, tvec = cv2.solvePnP(
        obj, img, intrinsics.matrix, intrinsics.distortion, flags=cv2.SOLVEPNP_SQPNP
    )
    if not ok:
        raise Contact3DError("손 자세(solvePnP) 추정에 실패했습니다.")

    error = _reprojection_error(obj, img, rvec, tvec, intrinsics)
    rotation, _ = cv2.Rodrigues(rvec)
    return Pose(rotation=rotation, translation=tvec.reshape(3)), error


@dataclass(frozen=True)
class ContactSample:
    """한 프레임의 3D 접촉 측정 결과."""

    raw_height_mm: float  # 영점 보정 전 측정 높이
    height_mm: float  # 영점 보정 후 유효 높이 (판정에 쓰는 값)
    plane_xy_mm: tuple[float, float]  # 손끝을 평면에 정사영한 좌표
    reprojection_error_px: float  # 손 자세 추정 품질 (클수록 신뢰도 낮음)
    tip_cam_mm: np.ndarray  # 손끝의 카메라 좌표(mm) — 디버그 시각화용


@dataclass(frozen=True)
class ContactDebugGeometry:
    """높이 판정 근거를 화면에 그리기 위한, 이미 이미지 픽셀로 투영된 기하 정보.

    ``core``는 그리지 않는다 — 좌표만 계산해 돌려주고, 실제 렌더링은 controller/ui가 한다.
    """

    tip_px: tuple[float, float]  # 손끝 (3D 자세로 재투영한 위치)
    foot_px: tuple[float, float]  # 손끝에서 평면에 내린 수선의 발
    #: 유효 높이 눈금: ``(높이mm, 픽셀좌표)`` 목록. 수선을 따라 찍어 척도를 보여준다.
    ruler_px: tuple[tuple[float, tuple[float, float]], ...]
    #: 접촉 문턱 눈금: ``(라벨, 높이mm, 픽셀좌표)``. 손끝이 문턱의 위/아래 어디인지 눈으로 확인.
    threshold_px: tuple[tuple[str, float, tuple[float, float]], ...]
    height_mm: float
    raw_height_mm: float
    plane_xy_mm: tuple[float, float]
    inside_plane: bool
    reprojection_error_px: float


class Contact3DEstimator:
    """평면 자세 + intrinsics를 들고, 프레임마다 손끝 높이(mm)를 계산한다."""

    def __init__(
        self,
        plane_pose: PlanePose,
        intrinsics: CameraIntrinsics,
        *,
        zero_offset_mm: float = 0.0,
        landmark_index: int = INDEX_TIP,
    ) -> None:
        self._plane = plane_pose
        self._intrinsics = intrinsics
        self._zero_offset_mm = float(zero_offset_mm)
        self._landmark_index = landmark_index

    @property
    def plane_pose(self) -> PlanePose:
        return self._plane

    @property
    def zero_offset_mm(self) -> float:
        return self._zero_offset_mm

    def set_zero_offset(self, offset_mm: float) -> None:
        """"표면에 닿았을 때의 측정 높이"를 영점으로 설정한다 (계통 오차 흡수)."""
        self._zero_offset_mm = float(offset_mm)

    def measure(self, hand) -> ContactSample | None:
        """``TrackedHand``에서 손끝 높이를 계산한다. world landmarks가 없으면 ``None``.

        자세 추정이 실패하거나 재투영 오차가 임계값을 넘으면 ``None``을 반환한다 —
        ``core`` 규약(예외 대신 빈 결과)을 따라 호출부가 폴백할 수 있게 한다.
        """
        world_mm = getattr(hand, "world_landmarks_mm", None)
        if world_mm is None:
            return None
        # 정수 픽셀로 반올림하면 그 양자화가 그대로 높이 오차가 되므로 서브픽셀 좌표를 우선한다.
        image_points = getattr(hand, "pixel_landmarks_f", None)
        if image_points is None:
            image_points = hand.pixel_landmarks
        try:
            hand_pose, error = estimate_hand_pose(world_mm, image_points, self._intrinsics)
        except (Contact3DError, cv2.error):
            return None
        if error > MAX_REPROJECTION_ERROR_PX:
            return None

        tip_cam = hand_pose.to_camera(world_mm[self._landmark_index].reshape(1, 3))[0]
        raw_height = self._plane.height_of(tip_cam)
        return ContactSample(
            raw_height_mm=raw_height,
            height_mm=raw_height - self._zero_offset_mm,
            plane_xy_mm=self._plane.plane_xy_of(tip_cam),
            reprojection_error_px=error,
            tip_cam_mm=tip_cam,
        )

    def project(self, points_cam: np.ndarray) -> np.ndarray:
        """카메라 좌표 ``(N, 3)`` mm → 이미지 픽셀 ``(N, 2)``.

        점이 이미 카메라 좌표계에 있으므로 회전·이동은 0을 넣는다.
        """
        pts = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
        zeros = np.zeros(3, dtype=np.float64)
        projected, _ = cv2.projectPoints(
            pts, zeros, zeros, self._intrinsics.matrix, self._intrinsics.distortion
        )
        return projected.reshape(-1, 2)

    def debug_geometry(
        self,
        sample: ContactSample,
        *,
        ruler_mm: tuple[float, ...] = DEFAULT_RULER_MM,
        thresholds: tuple[tuple[str, float], ...] = (),
    ) -> ContactDebugGeometry | None:
        """한 프레임의 높이 판정을 눈으로 볼 수 있도록 기하를 이미지 좌표로 투영한다.

        손끝에서 평면에 내린 **수선**(손끝 → 수선의 발)과, 그 수선을 따라 찍은 유효 높이
        눈금·접촉 문턱 눈금을 돌려준다. 눈금은 모두 **영점 보정 후 유효 높이** 기준이라
        ``ContactDetector``의 문턱과 같은 좌표계에서 비교할 수 있다.

        투영이 실패하면(자세가 카메라 뒤로 가는 등) ``None``.
        """
        foot_xy = sample.plane_xy_mm
        levels = tuple(float(v) for v in ruler_mm)
        thresh = tuple((str(label), float(mm)) for label, mm in thresholds)
        heights = levels + tuple(mm for _, mm in thresh)
        # 손끝 자체 + (수선의 발 포함) 모든 눈금 높이를 한 번에 투영한다.
        # 수선의 발은 **기하학적 평면**(원시 높이 0)이고, 눈금은 **유효 높이**(영점 보정 후)
        # 기준이라 원시 높이로 되돌려 투영한다. 둘의 간격이 곧 영점 보정량이라, 화면에서
        # 발과 0mm 눈금이 얼마나 떨어져 보이는지로 보정량 자체를 눈으로 확인할 수 있다.
        points = [sample.tip_cam_mm, self._plane.point_at(foot_xy, 0.0)] + [
            self._plane.point_at(foot_xy, mm + self._zero_offset_mm) for mm in heights
        ]
        try:
            pixels = self.project(np.asarray(points, dtype=np.float64))
        except cv2.error:
            return None
        if not np.all(np.isfinite(pixels)):
            return None

        tip_px = (float(pixels[0][0]), float(pixels[0][1]))
        foot_px = (float(pixels[1][0]), float(pixels[1][1]))
        rest = pixels[2:]
        ruler_px = tuple(
            (mm, (float(rest[i][0]), float(rest[i][1]))) for i, mm in enumerate(levels)
        )
        offset = len(levels)
        threshold_px = tuple(
            (label, mm, (float(rest[offset + i][0]), float(rest[offset + i][1])))
            for i, (label, mm) in enumerate(thresh)
        )
        return ContactDebugGeometry(
            tip_px=tip_px,
            foot_px=foot_px,
            ruler_px=ruler_px,
            threshold_px=threshold_px,
            height_mm=sample.height_mm,
            raw_height_mm=sample.raw_height_mm,
            plane_xy_mm=foot_xy,
            inside_plane=self._plane.contains_xy(foot_xy),
            reprojection_error_px=sample.reprojection_error_px,
        )


class ContactDetector:
    """유효 높이(mm)에 대한 히스테리시스 접촉 판정.

    ``pen_ratio``와 방향이 같다 — 값이 **작을수록** 접촉이다.
    """

    def __init__(
        self,
        *,
        down_mm: float = DEFAULT_CONTACT_DOWN_MM,
        up_mm: float = DEFAULT_CONTACT_UP_MM,
    ) -> None:
        if down_mm > up_mm:
            raise ValueError("down_mm must be <= up_mm for stable hysteresis")
        self.down_mm = float(down_mm)
        self.up_mm = float(up_mm)
        self._contact = False

    @property
    def contact(self) -> bool:
        return self._contact

    def update(self, height_mm: float) -> bool:
        if self._contact:
            self._contact = height_mm < self.up_mm
        else:
            self._contact = height_mm < self.down_mm
        return self._contact

    def set_thresholds(self, *, down_mm: float, up_mm: float) -> None:
        if down_mm > up_mm:
            raise ValueError("down_mm must be <= up_mm for stable hysteresis")
        self.down_mm = float(down_mm)
        self.up_mm = float(up_mm)

    def reset(self) -> None:
        self._contact = False
