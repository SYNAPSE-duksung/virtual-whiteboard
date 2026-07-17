"""거리 정규화 수식 검증 (3주차 A파트) — 합성 3D 손 모델 기반.

정답 라벨이 필요한 "판정 성능"(임계값·ROC)이 아니라, **수식 자체가 의도한 불변성을
갖는지**를 검증한다. 실촬영 데이터 없이 돌아가며, 랜드마크가 주어졌을 때 수식이
올바른가만 본다 (MediaPipe가 랜드마크를 잘 뽑는지는 검증 범위 밖 — 실촬영 영역).

검증 논리
---------
정규화 비율은 두 가지 요구를 동시에 만족해야 한다.

1. **신호(signal)는 살릴 것**: 손가락이 굽혀지면(pen down↔up) 비율이 크게 변해야 한다.
2. **잡음(spurious)은 죽일 것**: 손가락 자세가 그대로인데 손 전체가 회전(roll)했다고
   비율이 변하면 안 된다. 사선 구도에서 글씨를 쓰면 손 roll이 계속 바뀌므로, 이 변동은
   그대로 오판정이 된다.

그래서 핵심 지표는 **S/N = (굽힘에 의한 변화폭) / (roll에 의한 변화폭)** 이다. 클수록 좋다.

실행:
    python -m tools.validate_normalization              # 표 출력 + 그래프 저장
    python -m tools.validate_normalization --no-plot    # 표만
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np

from core.distances import (
    HAND_SCALE_METHODS,
    INDEX_DIP,
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_MCP,
    NUM_LANDMARKS,
    PINKY_MCP,
    RING_MCP,
    THUMB_CMC,
    THUMB_IP,
    THUMB_MCP,
    THUMB_TIP,
    WRIST,
    HandGeometry,
)

# ---------------------------------------------------------------------------
# 합성 손 모델 — 표준 성인 손 치수(mm), 손목 원점 / 손가락 +Y / 손바닥 XY평면
# ---------------------------------------------------------------------------
_CANONICAL_HAND_MM = np.zeros((NUM_LANDMARKS, 3), dtype=np.float64)
_CANONICAL_HAND_MM[WRIST] = (0, 0, 0)
# 엄지
_CANONICAL_HAND_MM[THUMB_CMC] = (-25, 25, 0)
_CANONICAL_HAND_MM[THUMB_MCP] = (-45, 50, 0)
_CANONICAL_HAND_MM[THUMB_IP] = (-58, 70, 0)
_CANONICAL_HAND_MM[THUMB_TIP] = (-68, 85, 0)
# 손바닥 MCP 열
_CANONICAL_HAND_MM[INDEX_MCP] = (-20, 90, 0)
_CANONICAL_HAND_MM[MIDDLE_MCP] = (0, 95, 0)
_CANONICAL_HAND_MM[RING_MCP] = (20, 92, 0)
_CANONICAL_HAND_MM[PINKY_MCP] = (38, 85, 0)
# 검지 / 중지 / 약지 / 새끼 (PIP, DIP, TIP)
_CANONICAL_HAND_MM[6:9] = [(-22, 135, 0), (-23, 160, 0), (-24, 180, 0)]
_CANONICAL_HAND_MM[10:13] = [(0, 145, 0), (0, 175, 0), (0, 197, 0)]
_CANONICAL_HAND_MM[14:17] = [(21, 137, 0), (22, 165, 0), (23, 185, 0)]
_CANONICAL_HAND_MM[18:21] = [(40, 120, 0), (41, 140, 0), (42, 158, 0)]


def _rot_x(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rot_y(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rot_z(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def make_hand(flex_deg: float = 0.0) -> np.ndarray:
    """검지를 ``flex_deg``만큼 굽힌 3D 손 (mm).

    PIP 관절에서 (DIP, TIP)을, DIP 관절에서 TIP을 각각 굽혀 2관절 굴곡을 흉내낸다.
    ``flex_deg=0``이면 쫙 편 상태, 값이 커질수록 손끝이 손바닥 쪽(-Z)으로 말린다.
    """
    hand = _CANONICAL_HAND_MM.copy()
    # PIP 기준으로 DIP, TIP 회전
    pivot = hand[INDEX_PIP].copy()
    rot = _rot_x(-flex_deg)
    for idx in (INDEX_DIP, INDEX_TIP):
        hand[idx] = rot @ (hand[idx] - pivot) + pivot
    # DIP 기준으로 TIP 추가 회전
    pivot = hand[INDEX_DIP].copy()
    rot = _rot_x(-flex_deg)
    hand[INDEX_TIP] = rot @ (hand[INDEX_TIP] - pivot) + pivot
    return hand


def project(
    hand_mm: np.ndarray,
    *,
    width: int = 1280,
    height: int = 720,
    focal_px: float = 1000.0,
    pitch_deg: float = 55.0,
    roll_deg: float = 0.0,
    yaw_deg: float = 0.0,
    distance_mm: float = 500.0,
    offset_mm: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """3D 손을 핀홀 카메라로 투영해 MediaPipe식 정규화 랜드마크 ``(21,3)``을 만든다.

    ``pitch_deg``는 책상을 내려다보는 사선 각도, ``roll_deg``는 카메라 광축을 중심으로 한
    손의 회전(글씨를 쓰며 손목이 돌아가는 성분)이다.

    z는 MediaPipe 규약(손목 기준 상대 깊이, x와 대략 같은 스케일)을 흉내내 채운다.
    """
    rotation = _rot_z(roll_deg) @ _rot_y(yaw_deg) @ _rot_x(pitch_deg)
    translation = np.array([offset_mm[0], offset_mm[1], distance_mm], dtype=np.float64)
    cam = hand_mm @ rotation.T + translation

    depth = cam[:, 2]
    if np.any(depth <= 1e-6):
        raise ValueError("손이 카메라 뒤로 넘어갔다 — distance_mm을 키울 것")

    nx = (focal_px * cam[:, 0] / depth + width / 2) / width
    ny = (focal_px * cam[:, 1] / depth + height / 2) / height
    # MediaPipe z: 손목 기준 상대 깊이를 x와 같은 스케일로.
    nz = (depth - depth[WRIST]) * focal_px / (depth[WRIST] * width)
    return np.column_stack([nx, ny, nz])


# ---------------------------------------------------------------------------
# 비교 대상 수식 (분자 × 분모)
# ---------------------------------------------------------------------------
def _num_ydiff_tip_dip(g: HandGeometry) -> float:
    """베이스라인: Tip–DIP의 **y좌표 차이만** (현행 compute_pen_ratio의 분자)."""
    return abs(g.points[INDEX_TIP][1] - g.points[INDEX_DIP][1])


def _num_euclid_tip_dip(g: HandGeometry) -> float:
    return g.distance(INDEX_TIP, INDEX_DIP)


def _num_euclid_tip_pip(g: HandGeometry) -> float:
    return g.distance(INDEX_TIP, INDEX_PIP)


def _num_euclid_tip_mcp(g: HandGeometry) -> float:
    return g.distance(INDEX_TIP, INDEX_MCP)


NUMERATORS = {
    "ydiff_tip_dip (베이스라인)": _num_ydiff_tip_dip,
    "euclid_tip_dip": _num_euclid_tip_dip,
    "euclid_tip_pip": _num_euclid_tip_pip,
    "euclid_tip_mcp": _num_euclid_tip_mcp,
}


@dataclass(frozen=True)
class RatioSpec:
    """분자 이름 + 분모(hand_scale) 방식 조합. 실데이터 ROC 스크립트에서도 재사용 가능."""

    numerator: str
    scale_method: str

    @property
    def label(self) -> str:
        return f"{self.numerator} / {self.scale_method}"

    def compute(self, landmarks: np.ndarray, width: int, height: int, *, use_z: bool = False) -> float:
        g = HandGeometry(landmarks, width, height, use_z=use_z)
        denominator = g.scale(self.scale_method)
        if denominator < 1e-6:
            return math.nan
        return NUMERATORS[self.numerator](g) / denominator


def _stats(values: list[float]) -> tuple[float, float, float]:
    """(평균, 변화폭 max-min, 변동계수 CV=std/mean)."""
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    spread = float(arr.max() - arr.min())
    cv = float(arr.std() / mean) if abs(mean) > 1e-12 else math.nan
    return mean, spread, cv


# ---------------------------------------------------------------------------
# 실험
# ---------------------------------------------------------------------------
W, H = 1280, 720
ROLL_RANGE = np.arange(-40, 41, 2.0)      # 손목이 돌아가는 현실적 범위 (광축 중심 회전)
PITCH_RANGE = np.arange(45, 66, 1.0)      # 글씨 쓰며 손 전체가 기우는 범위
FLEX_RANGE = np.arange(0, 61, 2.0)        # pen up(폄) ↔ pen down(굽힘)
_MID_FLEX = 30.0


def experiment_signal_vs_spurious(specs: list[RatioSpec]) -> list[dict]:
    """핵심 실험: roll에 의한 헛변동 대비 굽힘에 의한 신호 변화폭 (S/N)."""
    rows = []
    for spec in specs:
        # (1) 잡음: 손가락 자세 고정, roll만 변화 → 이상적으로는 변화 0이어야 함
        spurious = [
            spec.compute(project(make_hand(_MID_FLEX), width=W, height=H, roll_deg=r), W, H)
            for r in ROLL_RANGE
        ]
        # (2) 신호: roll 고정, 손가락 굽힘만 변화 → 크게 변해야 함
        signal = [
            spec.compute(project(make_hand(f), width=W, height=H, roll_deg=0.0), W, H)
            for f in FLEX_RANGE
        ]
        _, spur_spread, spur_cv = _stats(spurious)
        _, sig_spread, _ = _stats(signal)
        snr = sig_spread / spur_spread if spur_spread > 1e-12 else math.inf
        rows.append({
            "spec": spec,
            "signal_spread": sig_spread,
            "spurious_spread": spur_spread,
            "spurious_cv": spur_cv,
            "snr": snr,
        })
    return rows


def experiment_scale_stability() -> list[dict]:
    """분모(hand_scale) 5종이 카메라/손 회전에 얼마나 흔들리는가 (CV 낮을수록 안정)."""
    poses = [
        project(make_hand(_MID_FLEX), width=W, height=H, pitch_deg=p, roll_deg=r, yaw_deg=y)
        for p in (40.0, 55.0, 70.0)
        for r in (-30.0, 0.0, 30.0)
        for y in (-20.0, 0.0, 20.0)
    ]
    rows = []
    for method in HAND_SCALE_METHODS:
        values = [HandGeometry(lm, W, H).scale(method) for lm in poses]
        mean, spread, cv = _stats(values)
        rows.append({"method": method, "mean_px": mean, "spread_px": spread, "cv": cv})
    return rows


def experiment_invariance(specs: list[RatioSpec]) -> list[dict]:
    """카메라 거리·평행이동·해상도(종횡비)가 바뀌어도 비율이 유지되는가."""
    rows = []
    for spec in specs:
        base = spec.compute(project(make_hand(_MID_FLEX), width=W, height=H), W, H)

        dist_vals = [
            spec.compute(project(make_hand(_MID_FLEX), width=W, height=H, distance_mm=d), W, H)
            for d in (400.0, 500.0, 700.0, 1000.0)
        ]
        shift_vals = [
            spec.compute(
                project(make_hand(_MID_FLEX), width=W, height=H, offset_mm=(dx, dy)), W, H
            )
            for dx, dy in ((-80, -40), (0, 0), (80, 40))
        ]
        res_vals = [
            spec.compute(project(make_hand(_MID_FLEX), width=w, height=h), w, h)
            for w, h in ((640, 480), (1280, 720), (1920, 1080), (1000, 1000))
        ]
        rows.append({
            "spec": spec,
            "base": base,
            "distance_dev": max(abs(v - base) for v in dist_vals) / base * 100,
            "shift_dev": max(abs(v - base) for v in shift_vals) / base * 100,
            "aspect_dev": max(abs(v - base) for v in res_vals) / base * 100,
        })
    return rows


def experiment_noise(specs: list[RatioSpec], *, sigma: float = 0.003, trials: int = 500) -> list[dict]:
    """정규화 좌표에 가우시안 노이즈를 주입했을 때 비율이 얼마나 떨리는가."""
    rng = np.random.default_rng(42)
    clean = project(make_hand(_MID_FLEX), width=W, height=H)
    rows = []
    for spec in specs:
        base = spec.compute(clean, W, H)
        noisy = []
        for _ in range(trials):
            lm = clean.copy()
            lm[:, :2] += rng.normal(0.0, sigma, size=(NUM_LANDMARKS, 2))
            noisy.append(spec.compute(lm, W, H))
        arr = np.asarray(noisy)
        rows.append({
            "spec": spec,
            "base": base,
            "std": float(arr.std()),
            "rel_std": float(arr.std() / base * 100),
        })
    return rows


def _noise_std(spec: RatioSpec, *, sigma: float = 0.003, trials: int = 300) -> float:
    rng = np.random.default_rng(42)
    clean = project(make_hand(_MID_FLEX), width=W, height=H)
    values = []
    for _ in range(trials):
        lm = clean.copy()
        lm[:, :2] += rng.normal(0.0, sigma, size=(NUM_LANDMARKS, 2))
        values.append(spec.compute(lm, W, H))
    return float(np.std(values))


def experiment_grid() -> list[dict]:
    """분자 4종 × 분모 5종 = 20조합 종합 평가.

    헛변동은 두 축을 모두 본다. 손가락 자세는 고정한 채—
      * roll  : 광축 중심 회전 (손목이 돌아감)
      * pitch : 손 전체가 기움 (글씨를 쓰며 자연히 변함)
    둘 다 손가락 굽힘과 무관하므로, 비율이 변하면 그대로 오판정이다.
    pitch를 빼면 분모의 단축(foreshortening) 약점이 드러나지 않는다.

    최종 점수 = 신호폭 / (roll 헛변동 + pitch 헛변동 + 노이즈 표준편차)
    """
    rows = []
    for numerator in NUMERATORS:
        for method in HAND_SCALE_METHODS:
            spec = RatioSpec(numerator, method)
            roll_vals = [
                spec.compute(project(make_hand(_MID_FLEX), width=W, height=H, roll_deg=r), W, H)
                for r in ROLL_RANGE
            ]
            pitch_vals = [
                spec.compute(project(make_hand(_MID_FLEX), width=W, height=H, pitch_deg=p), W, H)
                for p in PITCH_RANGE
            ]
            signal = [
                spec.compute(project(make_hand(f), width=W, height=H), W, H)
                for f in FLEX_RANGE
            ]
            _, roll_spread, _ = _stats(roll_vals)
            _, pitch_spread, _ = _stats(pitch_vals)
            _, sig_spread, _ = _stats(signal)
            noise = _noise_std(spec)
            rows.append({
                "spec": spec,
                "signal": sig_spread,
                "roll": roll_spread,
                "pitch": pitch_spread,
                "noise": noise,
                "score": sig_spread / (roll_spread + pitch_spread + noise + 1e-12),
            })
    return rows


# ---------------------------------------------------------------------------
# 출력
# ---------------------------------------------------------------------------
def _print_header(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def run(plot: bool = True, plot_path: str = "output/normalization_validation.png") -> None:
    default_specs = [RatioSpec(n, "wrist_middle_mcp") for n in NUMERATORS]

    _print_header("[실험 1] 신호 대 헛변동  S/N = (굽힘 변화폭) / (roll 변화폭)   ※ 클수록 좋음")
    print("  roll(손목 회전)은 손가락 자세와 무관 → 비율이 변하면 그대로 오판정")
    rows = experiment_signal_vs_spurious(default_specs)
    print(f"\n  {'수식 (분모=wrist_middle_mcp)':<32}{'신호폭':>9}{'헛변동폭':>10}{'헛변동CV':>10}{'S/N':>9}")
    print(f"  {'-'*32}{'-'*9}{'-'*10}{'-'*10}{'-'*9}")
    for r in sorted(rows, key=lambda x: -x["snr"]):
        snr = "∞" if math.isinf(r["snr"]) else f"{r['snr']:.1f}"
        print(f"  {r['spec'].numerator:<32}{r['signal_spread']:>9.3f}{r['spurious_spread']:>10.3f}"
              f"{r['spurious_cv']*100:>9.1f}%{snr:>9}")

    _print_header("[실험 2] 분모(hand_scale) 회전 안정성   ※ CV 낮을수록 안정")
    print("  pitch 40/55/70° × roll ±30° × yaw ±20° = 27개 자세에서의 분모 변동")
    srows = experiment_scale_stability()
    print(f"\n  {'분모 방식':<20}{'평균(px)':>11}{'변화폭(px)':>12}{'CV':>9}")
    print(f"  {'-'*20}{'-'*11}{'-'*12}{'-'*9}")
    for r in sorted(srows, key=lambda x: x["cv"]):
        print(f"  {r['method']:<20}{r['mean_px']:>11.1f}{r['spread_px']:>12.1f}{r['cv']*100:>8.1f}%")

    _print_header("[실험 3] 불변성 — 카메라 거리 / 평행이동 / 해상도·종횡비   ※ 편차 낮을수록 좋음")
    irows = experiment_invariance(default_specs)
    print(f"\n  {'수식':<32}{'거리편차':>10}{'이동편차':>10}{'종횡비편차':>12}")
    print(f"  {'-'*32}{'-'*10}{'-'*10}{'-'*12}")
    for r in irows:
        print(f"  {r['spec'].numerator:<32}{r['distance_dev']:>9.2f}%{r['shift_dev']:>9.2f}%"
              f"{r['aspect_dev']:>11.2f}%")

    _print_header("[실험 4] 랜드마크 노이즈 견고성 (σ=0.003 정규화단위, 500회)   ※ 낮을수록 좋음")
    nrows = experiment_noise(default_specs)
    print(f"\n  {'수식':<32}{'기준값':>10}{'표준편차':>10}{'상대편차':>10}")
    print(f"  {'-'*32}{'-'*10}{'-'*10}{'-'*10}")
    for r in sorted(nrows, key=lambda x: x["rel_std"]):
        print(f"  {r['spec'].numerator:<32}{r['base']:>10.4f}{r['std']:>10.4f}{r['rel_std']:>9.2f}%")

    _print_header("[실험 5] 종합 — 분자 4종 × 분모 5종 = 20조합")
    print("  점수 = 신호폭 / (roll 헛변동 + pitch 헛변동 + 노이즈)   ※ 클수록 좋음")
    print("  roll·pitch 모두 손가락 자세는 고정한 채 손만 움직인 것 → 변하면 오판정")
    grid = sorted(experiment_grid(), key=lambda x: -x["score"])
    header = (f"\n  {'순위':<5}{'분자':<26}{'분모':<19}{'신호':>8}"
              f"{'roll':>8}{'pitch':>8}{'노이즈':>9}{'점수':>7}")
    print(header)
    print(f"  {'-'*5}{'-'*26}{'-'*19}{'-'*8}{'-'*8}{'-'*8}{'-'*9}{'-'*7}")
    for i, r in enumerate(grid[:8], 1):
        s = r["spec"]
        print(f"  {i:<5}{s.numerator.replace(' (베이스라인)',''):<26}{s.scale_method:<19}"
              f"{r['signal']:>8.3f}{r['roll']:>8.3f}{r['pitch']:>8.3f}"
              f"{r['noise']:>9.4f}{r['score']:>7.1f}")
    baseline = next(r for r in grid if r["spec"].numerator.startswith("ydiff")
                    and r["spec"].scale_method == "wrist_middle_mcp")
    rank = grid.index(baseline) + 1
    print(f"  ...\n  {rank:<5}{'ydiff_tip_dip':<26}{'wrist_middle_mcp':<19}"
          f"{baseline['signal']:>8.3f}{baseline['roll']:>8.3f}{baseline['pitch']:>8.3f}"
          f"{baseline['noise']:>9.4f}{baseline['score']:>7.1f}   ← 현행 베이스라인")

    best = grid[0]

    # 신호 곡선이 단조인가? 단조가 아니면 같은 비율이 두 굽힘각에 대응 → 임계값 자체가 모호해진다.
    best_curve = np.array(
        [best["spec"].compute(project(make_hand(f), width=W, height=H), W, H) for f in FLEX_RANGE]
    )
    peak_idx = int(np.argmax(best_curve))
    monotonic = peak_idx in (0, len(best_curve) - 1)
    peak_note = (
        "신호 곡선은 단조 → 임계값 1개로 분리 가능."
        if monotonic
        else (
            f"신호 곡선이 굽힘 {FLEX_RANGE[peak_idx]:.0f}°에서 꺾인다(단조 아님).\n"
            f"     → 같은 비율값이 서로 다른 굽힘각 2개에 대응하고, 꺾이는 지점 부근에서는\n"
            f"       기울기가 0에 가까워 감도가 사라진다. 단일 임계값 방식의 구조적 한계이며,\n"
            f"       실제 필기가 어느 굽힘 구간을 쓰는지는 실촬영 데이터로 확인해야 한다."
        )
    )

    _print_header("결론")
    print(f"""
  1. 현행 베이스라인(ydiff_tip_dip / wrist_middle_mcp)은 20조합 중 {rank}위.
     권장: 분자={best['spec'].numerator}, 분모={best['spec'].scale_method}
     (점수 {best['score']:.1f} vs 베이스라인 {baseline['score']:.1f})

  2. [분자] y좌표 차이가 문제다. 손목 roll이 ±40° 흔들리면 손가락 자세가 그대로여도
     비율이 {baseline['roll']:.3f} 변동한다 — 굽힘 신호폭({baseline['signal']:.3f})의 {baseline['roll']/baseline['signal']*100:.0f}%.
     사선 구도에서 글씨를 쓰면 roll은 계속 변하므로 그대로 오판정이 된다.
     유클리드 거리로 바꾸면 이 헛변동이 원리적으로 0이 된다 (회전은 거리를 보존).

  3. [분모] wrist_middle_mcp는 손가락 방향(+Y)과 거의 나란해서 pitch(내려다보는 각도)에
     그대로 단축된다 — 회전 안정성 5종 중 3위(CV 26.7%). palm_width는 pitch 축과 수직이라
     CV 2.7%로 약 10배 안정적이다. 위 표의 pitch 열에서 그 차이가 그대로 드러난다.

  4. 분자 개선(roll)과 분모 개선(pitch)은 **서로 다른 축의 문제**이므로 둘 다 고쳐야 한다.
     하나만 바꾸면 나머지 축의 헛변동이 그대로 남는다.

  5. [주의] {peak_note}

  ※ roll 헛변동 0.000은 "유클리드 거리는 회전 불변"이라는 수학적 성질이 파이프라인에서
    정확히 재현된 것이다. 실촬영에서는 MediaPipe 노이즈와 광축을 벗어난 회전 때문에 0이
    아니게 된다. 이 실험은 '수식'의 타당성만 검증하며, 임계값·정확도는 정답 라벨이 있어야 한다.
""")

    if plot:
        _save_plot(default_specs, plot_path)


def _save_plot(specs: list[RatioSpec], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    # 라벨은 영문 — matplotlib 기본 폰트에 한글 글리프가 없어 경고/두부가 뜬다.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for spec in specs:
        label = spec.numerator.replace(" (베이스라인)", " (baseline)")
        ax1.plot(
            ROLL_RANGE,
            [spec.compute(project(make_hand(_MID_FLEX), width=W, height=H, roll_deg=r), W, H)
             for r in ROLL_RANGE],
            label=label,
        )
        ax2.plot(
            FLEX_RANGE,
            [spec.compute(project(make_hand(f), width=W, height=H), W, H) for f in FLEX_RANGE],
            label=label,
        )
    ax1.set_title("Wrist roll sweep (finger pose FIXED)\nflat = good (no spurious variation)")
    ax1.set_xlabel("roll (deg)")
    ax1.set_ylabel("pen ratio")
    ax2.set_title("Finger flexion sweep (roll fixed)\nsteep = good (strong signal)")
    ax2.set_xlabel("finger flexion (deg)")
    ax2.set_ylabel("pen ratio")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Normalization formula validation (synthetic 3D hand, denominator=wrist_middle_mcp)")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    print(f"\n그래프 저장: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plot", action="store_true", help="그래프 저장 생략")
    parser.add_argument("--plot-path", default="output/normalization_validation.png")
    args = parser.parse_args()
    run(plot=not args.no_plot, plot_path=args.plot_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
