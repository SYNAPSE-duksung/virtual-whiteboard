"""D파트가 학습한 pen up/down 분류 모델(``scripts/train_pen_state.py``)의 실시간 추론 래퍼.

``scripts/train_pen_state.py``가 오프라인에서 CSV(``tools/extract_landmarks.py``가 남기는
``normalized_landmarks`` 원본, MediaPipe 정규화 좌표 0~1)로부터 계산하던 13개 kinematic
feature를 프레임 단위로 재현해 학습된 ``joblib`` 파이프라인(``StandardScaler`` +
LogisticRegression/SVM)에 넣는다. 반드시 ``normalized_landmarks``만 입력으로 써야 한다 —
``core.distances``의 픽셀 스케일링을 거치면 학습 시 분포와 달라져 예측이 무의미해진다.

``core/side_contact.py``와 같은 자체완결형 모듈 컨벤션을 따른다(다른 detector 모듈과
코드 공유 안 함, 4인 분업 병합 충돌 최소화). ``joblib``/``scikit-learn``은 이 모듈이
런타임 기본 의존성이 아니어도 ``core`` 패키지 자체는 깨지지 않도록 ``load()`` 안에서만
지연 import한다.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from core.distances import (
    INDEX_DIP,
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_MCP,
    WRIST,
)

DEFAULT_MODEL_PATH = Path("ml_results/logistic_regression.joblib")

#: 모델 입력 feature 순서. ``scripts/train_pen_state.py``의 ``feature_cols``와 정확히
#: 같아야 한다 — 순서가 어긋나면 학습된 스케일러/계수가 엉뚱한 값에 적용된다.
FEATURE_COLUMNS: tuple[str, ...] = (
    "tip_dip_ratio",
    "tip_pip_ratio",
    "tip_mcp_ratio",
    "dip_angle",
    "pip_angle",
    "tip_z_wrist",
    "tip_z_palm",
    "tip_y_wrist",
    "dip_y_wrist",
    "tip_dx",
    "tip_dy",
    "tip_dz",
    "tip_speed",
)

_EPS = 1e-8


def _distance(lm: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(lm[a] - lm[b]))


def _joint_angle(lm: np.ndarray, a: int, b: int, c: int) -> float:
    """``b``를 꼭짓점으로 ``a``, ``c``를 향하는 두 벡터 사이 각도(도)."""
    ba = lm[a] - lm[b]
    bc = lm[c] - lm[b]
    cosine = float(np.dot(ba, bc)) / (float(np.linalg.norm(ba)) * float(np.linalg.norm(bc)) + _EPS)
    return math.degrees(math.acos(min(1.0, max(-1.0, cosine))))


def compute_features(
    normalized_landmarks: np.ndarray, *, prev_tip_raw: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """``FEATURE_COLUMNS`` 순서의 feature 벡터와, 다음 프레임에 넘길 원시 tip 좌표를 반환한다.

    ``tip_dx/dy/dz``는 학습 스크립트와 동일하게 **원시(비정규화) 좌표의 차분을 현재
    프레임의 hand_scale로 나눈 값**이다(``(tip_raw[t] - tip_raw[t-1]) / hand_scale[t]``) —
    ``(tip_raw[t]/scale[t]) - (tip_raw[t-1]/scale[t-1])``이 아니므로, 캐시에는 스케일을
    나누기 전의 원시 좌표를 저장해야 한다. ``prev_tip_raw``가 없으면(추적 재시작 직후 등)
    속도 성분은 0으로 둔다.
    """
    lm = np.asarray(normalized_landmarks, dtype=np.float64)
    hand_scale = max(_distance(lm, WRIST, MIDDLE_MCP), _EPS)
    tip_raw = lm[INDEX_TIP].copy()

    if prev_tip_raw is None:
        tip_dx, tip_dy, tip_dz = 0.0, 0.0, 0.0
    else:
        delta = (tip_raw - prev_tip_raw) / hand_scale
        tip_dx, tip_dy, tip_dz = float(delta[0]), float(delta[1]), float(delta[2])

    features = np.array(
        [
            _distance(lm, INDEX_TIP, INDEX_DIP) / hand_scale,
            _distance(lm, INDEX_TIP, INDEX_PIP) / hand_scale,
            _distance(lm, INDEX_TIP, INDEX_MCP) / hand_scale,
            _joint_angle(lm, INDEX_PIP, INDEX_DIP, INDEX_TIP),
            _joint_angle(lm, INDEX_MCP, INDEX_PIP, INDEX_DIP),
            (lm[INDEX_TIP][2] - lm[WRIST][2]) / hand_scale,
            (lm[INDEX_TIP][2] - lm[MIDDLE_MCP][2]) / hand_scale,
            (lm[INDEX_TIP][1] - lm[WRIST][1]) / hand_scale,
            (lm[INDEX_DIP][1] - lm[WRIST][1]) / hand_scale,
            tip_dx,
            tip_dy,
            tip_dz,
            math.sqrt(tip_dx**2 + tip_dy**2 + tip_dz**2),
        ],
        dtype=np.float64,
    )
    return features, tip_raw


class MLPenStateDetector:
    """학습된 joblib 파이프라인(``StandardScaler`` + 분류기)으로 pen up/down을 예측한다."""

    def __init__(self, model, *, model_path: Path | None = None) -> None:
        self._model = model
        self.model_path = model_path
        self._prev_tip_raw: np.ndarray | None = None
        self._supports_proba = hasattr(model, "predict_proba")

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL_PATH) -> "MLPenStateDetector":
        """joblib 모델을 불러온다.

        언피클링은 실패 유형이 열려 있다(버전 불일치 시 ``AttributeError``, 라이브러리
        미설치 시 ``ModuleNotFoundError``, 손상 파일이면 ``EOFError`` 등) — 호출부가
        조용히 ``None``으로 폴백할 수 있도록 여기서는 잡지 않고 그대로 전파한다.
        """
        import joblib  # 지연 import: scikit-learn/joblib 없는 환경에서도 core는 임포트 가능해야 함

        model = joblib.load(Path(path))
        return cls(model, model_path=Path(path))

    @property
    def supports_confidence(self) -> bool:
        return self._supports_proba

    def update(self, normalized_landmarks: np.ndarray) -> tuple[bool, float | None]:
        """한 프레임을 예측한다. 반환은 ``(pen_down, confidence)`` — confidence는 모델이
        ``predict_proba``를 지원하지 않으면(예: ``probability=True`` 없이 학습된 SVM) None."""
        features, tip_raw = compute_features(normalized_landmarks, prev_tip_raw=self._prev_tip_raw)
        self._prev_tip_raw = tip_raw
        x = features.reshape(1, -1)
        pen_down = bool(int(self._model.predict(x)[0]))
        confidence = float(self._model.predict_proba(x)[0, 1]) if self._supports_proba else None
        return pen_down, confidence

    def reset(self) -> None:
        """트래킹 손실·지우기 제스처 시 호출: 프레임 간 속도 캐시를 비운다."""
        self._prev_tip_raw = None
