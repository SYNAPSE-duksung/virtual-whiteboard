"""실시간 좌표 스무딩 필터.

One Euro Filter (Casiez et al., 2012) — 저지연으로 손끝 좌표의 떨림(jitter)을 줄인다.
UI/CV 파이프라인과 독립적인 순수 수치 연산 모듈.
"""

from __future__ import annotations

import math


def _smoothing_factor(t_e: float, cutoff: float) -> float:
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def _exponential_smoothing(a: float, x: float, x_prev: float) -> float:
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    """1차원 One Euro Filter. x, y 좌표에 각각 하나씩 사용."""

    def __init__(
        self,
        t0: float,
        x0: float,
        dx0: float = 0.0,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def __call__(self, t: float, x: float) -> float:
        t_e = t - self.t_prev
        if t_e <= 0:
            t_e = 1e-3  # 동일 타임스탬프 방지

        a_d = _smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = _exponential_smoothing(a_d, dx, self.dx_prev)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _smoothing_factor(t_e, cutoff)
        x_hat = _exponential_smoothing(a, x, self.x_prev)

        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


class FingertipSmoother:
    """검지 끝 (x, y) 픽셀 좌표용 2D One Euro 스무더.

    손이 처음 감지될 때 내부 필터를 지연 초기화하고, 트래킹 손실 시 ``reset()``으로
    상태를 비운다(재검출 시 이전 좌표에서 이어지는 튐 방지).
    """

    def __init__(
        self,
        *,
        min_cutoff: float = 1.0,
        beta: float = 0.3,
        d_cutoff: float = 1.0,
    ) -> None:
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._fx: OneEuroFilter | None = None
        self._fy: OneEuroFilter | None = None

    def update(self, t: float, x: float, y: float) -> tuple[float, float]:
        """타임스탬프 ``t``의 원본 좌표를 스무딩해 반환. 첫 호출은 원본을 그대로 반환."""
        if self._fx is None or self._fy is None:
            self._fx = OneEuroFilter(
                t, x, min_cutoff=self._min_cutoff, beta=self._beta, d_cutoff=self._d_cutoff
            )
            self._fy = OneEuroFilter(
                t, y, min_cutoff=self._min_cutoff, beta=self._beta, d_cutoff=self._d_cutoff
            )
            return x, y
        return self._fx(t, x), self._fy(t, y)

    def reset(self) -> None:
        """트래킹 손실 시 호출: 다음 검출에서 필터를 새로 초기화한다."""
        self._fx = None
        self._fy = None
