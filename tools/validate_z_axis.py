"""Z축 임계값 실험 (3주차 A파트) — 합성 3D 손 모델 기반.

제안서의 pen-down 보조 판정 ②"MediaPipe z축 임계값"이 쓸 만한지 검증한다.
``tools/validate_normalization.py``의 손 모델·투영을 그대로 재사용한다.

⚠️ 이 실험의 결정적 한계
------------------------
합성 모델의 z는 **핀홀 카메라로 계산한 정확한 깊이**다. 반면 실제 MediaPipe의 z는 단안
RGB에서 **추정한 값**이라 훨씬 부정확하다. 따라서 이 실험은
    "z가 정확하다면 z 임계값이 쓸모 있는가?" (= 기하학적 상한)
만 답할 수 있고,
    "MediaPipe의 z가 실제로 정확한가?"
는 답할 수 없다 — 그건 실촬영 영역이다.

그래서 결론을 "z를 쓰자/말자"로 내지 않고, **"z 임계값이 성립하려면 z 노이즈가 얼마나
작아야 하는가"라는 요구사항(노이즈 예산)** 형태로 낸다. 실촬영 데이터로 실제 z 노이즈를
재서 이 예산과 비교하면 채택 여부가 결정된다.

실행:
    python -m tools.validate_z_axis
    python -m tools.validate_z_axis --no-plot
"""

from __future__ import annotations

import argparse

import numpy as np

from core.distances import INDEX_DIP, INDEX_TIP, WRIST, HandGeometry
from tools.validate_normalization import (
    FLEX_RANGE,
    PITCH_RANGE,
    ROLL_RANGE,
    H,
    W,
    _MID_FLEX,
    _stats,
    make_hand,
    project,
)

SIGMA_XY = 0.003          # xy 랜드마크 노이즈 (정규화 단위) — 검증2와 동일 기준
_TRIALS = 400
_SCALE_METHOD = "palm_width"   # 검증2에서 가장 안정적이었던 분모


# ---------------------------------------------------------------------------
# z 기반 지표 후보
# ---------------------------------------------------------------------------
def z_raw_tip(lm: np.ndarray) -> float:
    """손목 기준 손끝 z (픽셀 환산, 정규화 없음)."""
    p = HandGeometry(lm, W, H, use_z=True).points
    return p[INDEX_TIP][2] - p[WRIST][2]


def z_norm_tip(lm: np.ndarray) -> float:
    """손끝 z를 손 크기로 정규화."""
    g = HandGeometry(lm, W, H, use_z=True)
    scale = g.scale(_SCALE_METHOD)
    if scale < 1e-6:
        return np.nan
    return (g.points[INDEX_TIP][2] - g.points[WRIST][2]) / scale


def z_norm_tip_dip(lm: np.ndarray) -> float:
    """Tip–DIP의 z 차이를 손 크기로 정규화 (국소 지표)."""
    g = HandGeometry(lm, W, H, use_z=True)
    scale = g.scale(_SCALE_METHOD)
    if scale < 1e-6:
        return np.nan
    return (g.points[INDEX_TIP][2] - g.points[INDEX_DIP][2]) / scale


def dist_2d(lm: np.ndarray) -> float:
    """비교군: z를 안 쓰는 2D 유클리드 Tip–DIP / 손 크기 (검증2의 권장안)."""
    g = HandGeometry(lm, W, H, use_z=False)
    scale = g.scale(_SCALE_METHOD)
    return g.distance(INDEX_TIP, INDEX_DIP) / scale if scale > 1e-6 else np.nan


def dist_3d(lm: np.ndarray) -> float:
    """z를 포함한 3D 유클리드 Tip–DIP / 손 크기."""
    g = HandGeometry(lm, W, H, use_z=True)
    scale = g.scale(_SCALE_METHOD)
    return g.distance(INDEX_TIP, INDEX_DIP) / scale if scale > 1e-6 else np.nan


METRICS = {
    "z_raw_tip (정규화X)": z_raw_tip,
    "z_norm_tip": z_norm_tip,
    "z_norm_tip_dip": z_norm_tip_dip,
    "dist_2d (z 미사용, 비교군)": dist_2d,
    "dist_3d (z 포함)": dist_3d,
}


def _noisy(lm: np.ndarray, rng, sigma_xy: float, sigma_z: float) -> np.ndarray:
    out = lm.copy()
    out[:, :2] += rng.normal(0.0, sigma_xy, size=(lm.shape[0], 2))
    out[:, 2] += rng.normal(0.0, sigma_z, size=lm.shape[0])
    return out


def _noise_std(fn, sigma_z: float, *, sigma_xy: float = SIGMA_XY, trials: int = _TRIALS) -> float:
    rng = np.random.default_rng(11)
    clean = project(make_hand(_MID_FLEX), width=W, height=H)
    return float(np.std([fn(_noisy(clean, rng, sigma_xy, sigma_z)) for _ in range(trials)]))


# ---------------------------------------------------------------------------
# 실험
# ---------------------------------------------------------------------------
def experiment_signal_spurious() -> list[dict]:
    """z 지표가 굽힘에 반응하는가(신호), 손만 움직여도 변하는가(헛변동)."""
    rows = []
    for name, fn in METRICS.items():
        signal = [fn(project(make_hand(f), width=W, height=H)) for f in FLEX_RANGE]
        roll = [
            fn(project(make_hand(_MID_FLEX), width=W, height=H, roll_deg=r)) for r in ROLL_RANGE
        ]
        pitch = [
            fn(project(make_hand(_MID_FLEX), width=W, height=H, pitch_deg=p)) for p in PITCH_RANGE
        ]
        _, sig, _ = _stats(signal)
        _, roll_spread, _ = _stats(roll)
        _, pitch_spread, _ = _stats(pitch)
        rows.append({
            "name": name,
            "signal": sig,
            "roll": roll_spread,
            "pitch": pitch_spread,
        })
    return rows


def experiment_distance_invariance() -> list[dict]:
    """카메라 거리가 바뀌면 z 지표가 흔들리는가 (정규화 필요성 검증)."""
    distances = (400.0, 500.0, 700.0, 1000.0)
    rows = []
    for name, fn in METRICS.items():
        vals = [
            fn(project(make_hand(_MID_FLEX), width=W, height=H, distance_mm=d)) for d in distances
        ]
        base = vals[1]
        dev = max(abs(v - base) for v in vals) / abs(base) * 100 if abs(base) > 1e-9 else np.nan
        rows.append({"name": name, "values": vals, "dev_pct": dev})
    return rows


def experiment_noise_budget(multiples: np.ndarray) -> list[dict]:
    """z 노이즈를 xy 노이즈의 k배로 키우며 판별력(신호/노이즈)이 어떻게 무너지는가.

    MediaPipe의 실제 z 노이즈 크기를 모르므로, 절대값 대신 **xy 노이즈 대비 배수**로
    요구사항을 표현한다. 실촬영에서 z 노이즈를 재어 이 배수와 비교하면 된다.
    """
    rows = []
    for name in ("z_norm_tip", "z_norm_tip_dip", "dist_3d (z 포함)", "dist_2d (z 미사용, 비교군)"):
        fn = METRICS[name]
        signal = [fn(project(make_hand(f), width=W, height=H)) for f in FLEX_RANGE]
        _, sig, _ = _stats(signal)
        scores = []
        for k in multiples:
            noise = _noise_std(fn, SIGMA_XY * k)
            scores.append(sig / noise if noise > 1e-12 else np.inf)
        rows.append({"name": name, "signal": sig, "scores": scores})
    return rows


# ---------------------------------------------------------------------------
# 출력
# ---------------------------------------------------------------------------
def _hdr(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def run(plot: bool = True, plot_path: str = "output/z_axis_validation.png") -> None:
    _hdr("[실험 1] z 지표의 신호 vs 헛변동   ※ 신호 크고 헛변동 작아야 좋음")
    print("  신호=굽힘 0~60° 변화폭 / 헛변동=손가락 고정한 채 roll·pitch만 변화")
    rows = experiment_signal_spurious()
    print(f"\n  {'지표':<28}{'신호':>9}{'roll':>9}{'pitch':>9}{'신호/헛변동':>12}")
    print(f"  {'-'*28}{'-'*9}{'-'*9}{'-'*9}{'-'*12}")
    for r in rows:
        spur = r["roll"] + r["pitch"]
        ratio = r["signal"] / spur if spur > 1e-12 else np.inf
        ratio_s = "∞" if np.isinf(ratio) else f"{ratio:.2f}"
        print(f"  {r['name']:<28}{r['signal']:>9.3f}{r['roll']:>9.3f}"
              f"{r['pitch']:>9.3f}{ratio_s:>12}")

    _hdr("[실험 2] 카메라 거리 불변성 (400/500/700/1000mm)   ※ 정규화 필요성")
    drows = experiment_distance_invariance()
    print(f"\n  {'지표':<28}{'400mm':>10}{'500mm':>10}{'700mm':>10}{'1000mm':>10}{'편차':>9}")
    print(f"  {'-'*28}{'-'*10}{'-'*10}{'-'*10}{'-'*10}{'-'*9}")
    for r in drows:
        v = r["values"]
        print(f"  {r['name']:<28}{v[0]:>10.3f}{v[1]:>10.3f}{v[2]:>10.3f}{v[3]:>10.3f}"
              f"{r['dev_pct']:>8.1f}%")

    _hdr("[실험 3] z 노이즈 예산 — 판별력(신호/노이즈)이 언제 무너지는가")
    print(f"  z 노이즈를 xy 노이즈(σ={SIGMA_XY})의 k배로 키우며 측정")
    multiples = np.array([0.5, 1, 2, 3, 5, 8, 12, 20])
    nrows = experiment_noise_budget(multiples)
    head = "  " + f"{'지표':<28}" + "".join(f"{f'{k:g}x':>8}" for k in multiples)
    print(f"\n{head}")
    print(f"  {'-'*28}{'-'*8*len(multiples)}")
    for r in nrows:
        cells = "".join(f"{s:>8.1f}" for s in r["scores"])
        print(f"  {r['name']:<28}{cells}")

    baseline_2d = next(r for r in nrows if r["name"].startswith("dist_2d"))["scores"][0]
    d3 = next(r for r in nrows if r["name"].startswith("dist_3d"))
    ztd = next(r for r in nrows if r["name"] == "z_norm_tip_dip")

    def _crossover(scores) -> str:
        """판별력이 2D 비교군 아래로 떨어지는 첫 배수."""
        for k, s in zip(multiples, scores):
            if s < baseline_2d:
                return f"{k:g}x"
        return f">{multiples[-1]:g}x"

    sig_2d = next(r["signal"] for r in rows if r["name"].startswith("dist_2d"))
    sig_3d = next(r["signal"] for r in rows if r["name"].startswith("dist_3d"))
    ztd_pitch = next(r["pitch"] for r in rows if r["name"] == "z_norm_tip_dip")
    ztd_sig = next(r["signal"] for r in rows if r["name"] == "z_norm_tip_dip")

    _hdr("결론")
    print(f"""
  1. [정규화 필수] 정규화 안 한 z_raw_tip은 카메라 거리만 바뀌어도 {drows[0]['dev_pct']:.0f}% 흔들린다.
     z는 손목 기준 상대 깊이이고 x와 같은 스케일이라 원근에 그대로 딸려간다.
     **z 임계값을 절대값으로 박으면 안 된다** — 반드시 손 크기로 정규화할 것
     (정규화 시 편차 {drows[1]['dev_pct']:.1f}%).

  2. [dist_3d는 근본적으로 틀린 접근 — 폐기] 유클리드 거리에 z를 더하면 신호가 오히려
     **줄어든다** ({sig_2d:.3f} → {sig_3d:.3f}). 이유: **Tip–DIP의 3D 거리는 손가락 끝마디뼈 길이,
     즉 강체라서 굽혀도 변하지 않는다.** 2D 투영 거리가 신호를 갖는 건 오직 단축
     (foreshortening) 때문인데, z를 넣으면 그 단축을 복원해버려 신호를 스스로 지운다.
     실제로 dist_3d는 z가 거의 정확한 0.5x에서조차 2D보다 나쁘다({d3['scores'][0]:.1f} < {baseline_2d:.1f}).
     → **z를 거리 계산에 섞는 방식은 노이즈와 무관하게 기각.**

  3. [z 차분(z_norm_tip_dip)은 유망] 반면 z를 '거리'가 아니라 **깊이 차분** 자체로 쓰면
     신호 {ztd_sig:.3f}, pitch 헛변동 {ztd_pitch:.3f}으로 z_norm_tip(전체 깊이)보다 훨씬 낫다.
     z가 정확하면 판별력 {ztd['scores'][1]:.1f}(1x)로 2D 비교군({baseline_2d:.1f})을 앞선다.

  4. [핵심 — z 노이즈 예산] 단, z_norm_tip_dip의 판별력은 z 노이즈가 xy의
     **{_crossover(ztd['scores'])}를 넘으면 2D보다 못해진다.**
     → 요구사항: **MediaPipe z 노이즈 < xy 노이즈 × {_crossover(ztd['scores'])}** 일 때만 보조 판정 ②를 채택할 가치가 있다.

  5. [다음 단계 — 실측] 단안 RGB에서 추정하는 MediaPipe hand z가 실제로 이 예산 안에
     들어오는지는 **이 실험으로 답할 수 없다**(합성 z는 정확한 값이므로).
     → 정지 상태 손을 몇 초 촬영해 **z의 프레임간 표준편차 ÷ xy의 표준편차**를 재면 배수가
        바로 나온다. 그 값이 {_crossover(ztd['scores'])}를 넘으면 제안서의 보조 판정 ②는 폐기하는 게 맞다.
        (이 측정은 라벨링이 필요 없어 녹화만 되면 즉시 가능하다.)

  ※ 합성 z는 핀홀 카메라로 계산한 '정확한' 깊이다. 위 수치는 전부 'z가 정확하다면'의
    상한이며, 실제 MediaPipe z의 유용성은 이 실험으로 답할 수 없다.
""")

    if plot:
        _save_plot(multiples, nrows, baseline_2d, plot_path)


def _save_plot(multiples, nrows, baseline_2d, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from pathlib import Path

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for name, fn in METRICS.items():
        label = name.split(" ")[0]
        ax1.plot(
            FLEX_RANGE,
            [fn(project(make_hand(f), width=W, height=H)) for f in FLEX_RANGE],
            label=label,
        )
    ax1.set_title("Finger flexion sweep\n(does z respond to pen up/down?)")
    ax1.set_xlabel("finger flexion (deg)")
    ax1.set_ylabel("metric value")

    for r in nrows:
        ax2.plot(multiples, r["scores"], marker="o", label=r["name"].split(" ")[0])
    ax2.axhline(baseline_2d, ls="--", c="gray", lw=1, label="2D @ clean z")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_title("Z noise budget\ndiscriminability vs z-noise (xy-noise multiples)")
    ax2.set_xlabel("z noise / xy noise")
    ax2.set_ylabel("signal / noise  (higher=better)")

    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Z-axis threshold validation (synthetic 3D hand — z is EXACT here, unlike real MediaPipe)")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    print(f"\n그래프 저장: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--plot-path", default="output/z_axis_validation.png")
    args = parser.parse_args()
    run(plot=not args.no_plot, plot_path=args.plot_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
