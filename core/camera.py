"""카메라 내부 파라미터(intrinsics) — 3D 접촉 판정의 전제 조건.

2D 이미지 좌표에서 3차원 위치를 복원하려면 초점거리·주점을 알아야 한다
(``cv2.solvePnP``의 ``cameraMatrix``). 이 모듈은 그 값을 담고, 저장·로드하며,
정식 체스보드 캘리브레이션이 없을 때 쓸 **근사값**을 만들어 준다.

정확도에 대해
------------
- **정식 체스보드 캘리브레이션**이 가장 정확하다(``cv2.calibrateCamera``).
- 없을 때는 ``from_fov()``로 화각을 가정해 근사한다. 웹캠 수평 화각은 보통 55~70°이며,
  기본값 60°는 그 중간값이다. 초점거리가 20% 틀리면 복원된 **거리도 약 20% 비례해서**
  틀린다 — 다만 접촉 판정은 "표면에 닿았을 때의 높이 = 0"을 기준으로 **상대 변화**를
  보므로(``core.contact3d``의 영점 보정), 이 비례 오차는 상당 부분 흡수된다.
- 즉 근사 intrinsics로도 접촉 판정은 동작하지만, 높이의 **절대 mm 값**을 신뢰하려면
  체스보드 캘리브레이션이 필요하다.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_INTRINSICS_PATH = Path("output/camera_intrinsics.json")

# 일반적인 노트북/USB 웹캠의 수평 화각(도). from_fov()의 기본 가정값.
DEFAULT_HORIZONTAL_FOV_DEG = 60.0


@dataclass(frozen=True)
class CameraIntrinsics:
    """핀홀 카메라 내부 파라미터. 단위는 픽셀."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    # 렌즈 왜곡 계수 (k1, k2, p1, p2, k3). 근사 intrinsics에서는 0으로 둔다.
    dist_coeffs: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    # 체스보드 캘리브레이션 결과인지, 화각 가정으로 만든 근사값인지 구분한다.
    approximate: bool = True

    @classmethod
    def from_fov(
        cls,
        width: int,
        height: int,
        *,
        horizontal_fov_deg: float = DEFAULT_HORIZONTAL_FOV_DEG,
    ) -> "CameraIntrinsics":
        """수평 화각 가정으로 근사 intrinsics를 만든다 (정사각 픽셀·주점=이미지 중심 가정)."""
        if width <= 0 or height <= 0:
            raise ValueError("width/height는 양수여야 합니다")
        if not 10.0 < horizontal_fov_deg < 170.0:
            raise ValueError(f"비현실적인 화각입니다: {horizontal_fov_deg}")
        fx = (width / 2.0) / math.tan(math.radians(horizontal_fov_deg) / 2.0)
        return cls(fx=fx, fy=fx, cx=width / 2.0, cy=height / 2.0,
                   width=width, height=height, approximate=True)

    @property
    def matrix(self) -> np.ndarray:
        """``cv2.solvePnP``에 넘길 3x3 카메라 행렬."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def distortion(self) -> np.ndarray:
        return np.asarray(self.dist_coeffs, dtype=np.float64).reshape(-1, 1)

    def scaled_to(self, width: int, height: int) -> "CameraIntrinsics":
        """다른 해상도로 스케일한 intrinsics를 반환한다.

        캘리브레이션 당시와 실행 시 카메라 해상도가 다르면 초점거리·주점을 픽셀 비율만큼
        조정해야 한다 — 안 하면 복원 거리가 그 비율만큼 틀어진다.
        """
        if width <= 0 or height <= 0:
            raise ValueError("width/height는 양수여야 합니다")
        if (width, height) == (self.width, self.height):
            return self
        sx, sy = width / self.width, height / self.height
        return CameraIntrinsics(
            fx=self.fx * sx, fy=self.fy * sy, cx=self.cx * sx, cy=self.cy * sy,
            width=width, height=height, dist_coeffs=self.dist_coeffs,
            approximate=self.approximate,
        )

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
            "width": self.width, "height": self.height,
            "dist_coeffs": list(self.dist_coeffs),
            "approximate": self.approximate,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CameraIntrinsics":
        return cls(
            fx=float(data["fx"]), fy=float(data["fy"]),
            cx=float(data["cx"]), cy=float(data["cy"]),
            width=int(data["width"]), height=int(data["height"]),
            dist_coeffs=tuple(float(v) for v in data.get("dist_coeffs", (0.0,) * 5)),
            approximate=bool(data.get("approximate", True)),
        )

    def save(self, path: str | Path = DEFAULT_INTRINSICS_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path = DEFAULT_INTRINSICS_PATH) -> "CameraIntrinsics":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def load_or_estimate(
        cls,
        width: int,
        height: int,
        *,
        path: str | Path | None = DEFAULT_INTRINSICS_PATH,
        horizontal_fov_deg: float = DEFAULT_HORIZONTAL_FOV_DEG,
    ) -> "CameraIntrinsics":
        """저장된 캘리브레이션이 있으면 (해상도에 맞춰 스케일해서) 쓰고, 없으면 근사한다."""
        if path is not None:
            try:
                return cls.load(path).scaled_to(width, height)
            except (FileNotFoundError, OSError, ValueError, KeyError, TypeError):
                pass
        return cls.from_fov(width, height, horizontal_fov_deg=horizontal_fov_deg)
