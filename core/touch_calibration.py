"""터치(pen-down/up) 임계값 캘리브레이션 — 실사용자 hover/touch 표본으로 재추정.

``core.geometry``의 4점 캘리브레이션이 "어디"(필기 영역의 XY 범위)를 담당한다면, 이
모듈은 "언제"(접촉 여부)를 담당한다. 정확히는 여전히 ``compute_pen_ratio``(손가락 굽힘
정도) 하나의 신호에 의존하지만, 기본 임계값(0.55/0.70)은 임의의 손·카메라 각도에 맞춘
값일 뿐이다. 이 모듈은 **이 사람·이 카메라 구도에서 실제로 측정한** hover(화면 위에
띄운 채 안 닿음) 표본과 touch(눌러서 유지) 표본으로부터 히스테리시스 임계값 쌍을
다시 추정한다.

⚠️ 한계: 여전히 pen_ratio(2D 굽힘 신호) 기반이며, 진짜 3D 접촉 판정으로 바뀌는 것은
아니다 — 사람마다 다른 "얼마나 굽혀야 눌린 걸로 볼지" 감도를 이 사람 데이터로 맞출 뿐이다.
touch 단계에서 실제로 계속 눌러야 하는데, 중간에 살짝 뜨는 순간이 섞이면(자유롭게
"+"·"X"를 그리다 보면 생기기 쉽다) touch 표본에 hover 값이 섞여 추정이 나빠진다 — 그래서
자유 드로잉보다 **누른 채 정지 유지**를 권장한다(``controller/main.py``의 안내 문구 참고).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

DEFAULT_SAMPLE_SECONDS = 2.5
MIN_SAMPLES = 15  # 이보다 적으면 중앙값이 노이즈에 흔들려 추정을 거부한다.
_MARGIN_FRACTION = 0.18  # 두 클러스터 중앙값 간격의 몇 %를 히스테리시스 폭으로 뺄지.


class TouchCalibrationError(ValueError):
    """표본이 부족하거나 두 단계가 실제로 구분되지 않아 임계값을 추정할 수 없을 때."""


@dataclass
class TouchSampleCollector:
    """한 단계(hover 또는 touch)의 pen_ratio 표본을 모으는 단순 누산기."""

    samples: list[float] = field(default_factory=list)

    def add(self, pen_ratio: float | None) -> None:
        """``None``(손 미검출 프레임)은 무시한다."""
        if pen_ratio is not None:
            self.samples.append(pen_ratio)

    def reset(self) -> None:
        self.samples.clear()

    @property
    def count(self) -> int:
        return len(self.samples)


@dataclass(frozen=True)
class TouchThresholds:
    """추정된 임계값 쌍과, 어떻게 나왔는지 확인할 수 있는 근거 통계."""

    down_thresh: float
    up_thresh: float
    hover_median: float
    touch_median: float
    hover_count: int
    touch_count: int


def estimate_thresholds(
    hover_samples: list[float],
    touch_samples: list[float],
    *,
    margin_fraction: float = _MARGIN_FRACTION,
) -> TouchThresholds:
    """hover(비접촉) / touch(접촉) pen_ratio 표본으로부터 히스테리시스 임계값 쌍을 추정한다.

    ``compute_pen_ratio``는 값이 작을수록 눌린 상태를 뜻하므로
    ``touch_median < hover_median``이어야 한다 — 그렇지 않으면 두 단계가 실제로는
    구분되지 않았다는 뜻(예: touch 단계에서 실제로 누르지 않음)으로 간주해 거부한다.

    두 클러스터 중앙값의 중점을 경계로 잡고, 그 간격의 ``margin_fraction``만큼을 위아래로
    벌려 히스테리시스 폭을 만든다(채터링 방지). 중앙값을 쓰는 이유는 순간적인 트래킹
    노이즈 몇 프레임에 흔들리지 않기 위함이다.
    """
    # 이 메시지는 controller/main.py의 cv2 오버레이에 그대로 표시된다 — Hershey 폰트에
    # 한글 글리프가 없어 ASCII로만 작성한다.
    if len(hover_samples) < MIN_SAMPLES:
        raise TouchCalibrationError(
            f"not enough hover samples ({len(hover_samples)}/{MIN_SAMPLES}) - "
            "hold your finger above the surface longer."
        )
    if len(touch_samples) < MIN_SAMPLES:
        raise TouchCalibrationError(
            f"not enough touch samples ({len(touch_samples)}/{MIN_SAMPLES}) - "
            "hold your finger pressed on the surface longer."
        )

    hover_median = statistics.median(hover_samples)
    touch_median = statistics.median(touch_samples)
    if touch_median >= hover_median:
        raise TouchCalibrationError(
            "touch/hover samples are not separated "
            f"(touch median={touch_median:.3f} >= hover median={hover_median:.3f}) - "
            "check the finger didn't touch during hover, and did press during touch."
        )

    gap = hover_median - touch_median
    margin = gap * margin_fraction
    boundary = (hover_median + touch_median) / 2
    down_thresh = boundary - margin
    up_thresh = boundary + margin
    if up_thresh <= down_thresh:
        # 극단적으로 좁은 gap이면 margin이 0에 가까워질 수 있다 — 최소 폭을 보장.
        down_thresh, up_thresh = boundary, boundary + 1e-3

    # NOTE: 예전에는 down_thresh를 0으로 클램프했는데, 이 함수는 pen_ratio뿐 아니라
    # **높이(mm)** 표본에도 쓰인다(3D 접촉 캘리브레이션). 높이는 음수가 정상이므로
    # 클램프하면 문턱이 통째로 왜곡된다. pen_ratio처럼 입력이 모두 0 이상이면
    # down = 0.32*hover + 0.68*touch >= 0 이라 클램프는 애초에 발동하지 않는다.
    return TouchThresholds(
        down_thresh=down_thresh,
        up_thresh=up_thresh,
        hover_median=hover_median,
        touch_median=touch_median,
        hover_count=len(hover_samples),
        touch_count=len(touch_samples),
    )
