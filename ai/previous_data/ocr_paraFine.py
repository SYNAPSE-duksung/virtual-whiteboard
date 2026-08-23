"""
EasyOCR 손글씨 인식 파라미터 튜닝 스크립트
- 전처리 옵션 비교
- detect/recognize 파라미터 그리드 서치
- 결과를 CSV로 저장해서 어떤 조합이 가장 잘 맞는지 비교

사용법:
1. `IMAGE_DIR`에 검사할 이미지들이 들어있는 폴더 경로를 지정한다. (하위 폴더까지 자동 스캔)
2. (선택) ground_truth.json 파일을 `{"파일명.jpg": "정답텍스트", ...}` 형식으로 만들어두면
   CER(문자 오류율)까지 계산해서 비교 가능. 없어도 confidence 기준으로 비교됨.
3. 실행하면 results.csv 에 이미지 x 전처리 x 파라미터 조합별 결과가 전부 저장됨.

주의: 튜닝 단계는 MAX_TUNING_IMAGES(기본 3)장만 사용하며,
     전처리 5종 x detect 조합(2*2=4) x recognize 조합(2*2=4) = 이미지당 80번,
     3장이면 총 240번 정도로 CPU에서도 몇 분 내로 끝나는 수준.
     최적 조합을 찾은 뒤에는 그 조합 하나만으로 전체 폴더에 적용하면 됨.
"""

import cv2
import numpy as np
import easyocr
import itertools
import csv
import time
import json
from pathlib import Path

# ------------------------------------------------------------------
# 0. 설정
# ------------------------------------------------------------------
IMAGE_DIR = "./enhanced_v2"                 # 검사할 이미지들이 들어있는 폴더
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
RECURSIVE = True                       # 하위 폴더까지 재귀적으로 검사할지 여부

# 정답 텍스트 (선택). 파일명(확장자 제외 없이) -> 정답 텍스트
# 아래처럼 직접 채우거나, ground_truth.json 파일을 읽어와도 됨.
GROUND_TRUTH_JSON = " "  # {"sample1.jpg": "정답텍스트", ...} 형식, 없으면 빈 dict로 처리

LANGS = ["ko", "en"]  # 필요한 언어 코드로 변경
GPU = False  # CPU 환경이면 반드시 False로. True로 두면 매번 GPU 탐색 시도로 더 느려짐

# 튜닝(그리드 서치) 단계에서 쓸 이미지 수 제한 (전체 폴더가 아니라 일부만)
# CPU에서는 그리드 서치가 매우 느리므로, 먼저 소수 이미지로 최적 조합을 찾는 걸 권장
MAX_TUNING_IMAGES = 3


def collect_images(image_dir: str, recursive: bool = True) -> list[str]:
    """폴더 안의 이미지 파일 경로를 전부 수집"""
    root = Path(image_dir)
    if not root.exists():
        raise FileNotFoundError(f"이미지 폴더를 찾을 수 없습니다: {image_dir}")

    pattern = "**/*" if recursive else "*"
    paths = [
        str(p) for p in root.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    paths.sort()
    if not paths:
        print(f"경고: {image_dir} 에서 이미지 파일을 찾지 못했습니다.")
    else:
        print(f"이미지 {len(paths)}개 발견됨.")
    return paths


def load_ground_truth(json_path: str) -> dict:
    """정답 텍스트 JSON 로드 (없으면 빈 dict 반환)"""
    p = Path(json_path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

# ------------------------------------------------------------------
# 1. 전처리 함수들
# ------------------------------------------------------------------
def preprocess_variants(image_path):
    """손글씨 OCR용 최소 전처리 3종"""
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    variants = {}

    # 1. 원본 그레이스케일
    # 가장 기본적인 기준값
    variants["gray"] = gray

    # 2. CLAHE
    # 글씨와 배경의 대비를 조금 강화
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    variants["clahe"] = clahe.apply(gray)

    # 3. Adaptive Threshold
    # 배경 밝기가 일정하지 않은 이미지에 대응
    variants["adaptive_thresh"] = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15
    )

    return variants


# ------------------------------------------------------------------
# 2. EasyOCR 파라미터 그리드
# ------------------------------------------------------------------
# detect 단계 (2 x 2 = 4 조합)
DETECT_PARAM_GRID = {
    "text_threshold": [0.5, 0.7],        # 낮을수록 더 많은 영역을 텍스트로 인식 (기본 0.7)
    "mag_ratio": [1.0, 1.8],             # 이미지 확대 비율, 작은 글씨에 유효
}

# recognize 단계 (2 x 2 = 4 조합)
RECOGNIZE_PARAM_GRID = {
    "contrast_ths": [0.1, 0.3],          # 이 값보다 대비가 낮으면 adjust_contrast 적용
    "decoder": ["greedy", "beamsearch"], # beamsearch가 보통 더 정확하지만 느림
}


def cer(ref: str, hyp: str) -> float:
    """문자 오류율 (Character Error Rate), 간단한 Levenshtein 기반"""
    if not ref:
        return -1.0  # 정답 없음
    m, n = len(ref), len(hyp)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n] / max(m, 1)


# ------------------------------------------------------------------
# 3. 그리드 서치 실행
# ------------------------------------------------------------------
def run_grid_search():
    reader = easyocr.Reader(LANGS, gpu=GPU)

    all_image_paths = collect_images(IMAGE_DIR, RECURSIVE)
    ground_truth = load_ground_truth(GROUND_TRUTH_JSON)

    # 그리드 서치는 일부 이미지로만 (전체 폴더 X) - CPU에서 감당 가능한 수준으로
    image_paths = all_image_paths[:MAX_TUNING_IMAGES]
    if len(all_image_paths) > MAX_TUNING_IMAGES:
        print(f"전체 {len(all_image_paths)}장 중 튜닝용으로 {MAX_TUNING_IMAGES}장만 사용합니다.")
        print(f"(MAX_TUNING_IMAGES 값을 바꾸면 조절 가능. 나머지 이미지는 최적 조합 찾은 뒤 일괄 적용 권장)")

    detect_keys = list(DETECT_PARAM_GRID.keys())
    detect_combos = list(itertools.product(*DETECT_PARAM_GRID.values()))

    recog_keys = list(RECOGNIZE_PARAM_GRID.keys())
    recog_combos = list(itertools.product(*RECOGNIZE_PARAM_GRID.values()))

    total_calls = len(image_paths) * 3 * len(detect_combos) * len(recog_combos)
    print(f"총 {total_calls}번의 OCR 호출 예정.")

    results = []
    call_count = 0
    t_all_start = time.time()

    for idx, image_path in enumerate(image_paths, 1):
        print(f"[{idx}/{len(image_paths)}] 처리 중: {image_path}")
        variants = preprocess_variants(image_path)
        # 정답 매칭은 파일명(확장자 포함) 또는 stem(확장자 제외) 둘 다 시도
        fname = Path(image_path).name
        stem = Path(image_path).stem
        gt = ground_truth.get(fname, ground_truth.get(stem, ""))

        for prep_name, prep_img in variants.items():
            for d_combo in detect_combos:
                d_params = dict(zip(detect_keys, d_combo))
                for r_combo in recog_combos:
                    r_params = dict(zip(recog_keys, r_combo))

                    start = time.time()
                    try:
                        result = reader.readtext(
                            prep_img,
                            detail=1,
                            paragraph=False,
                            **d_params,
                            **r_params,
                        )
                    except Exception as e:
                        print(f"에러 발생 ({prep_name}, {d_params}, {r_params}): {e}")
                        continue
                    elapsed = time.time() - start

                    call_count += 1
                    if call_count % 10 == 0 or call_count == total_calls:
                        avg_per_call = (time.time() - t_all_start) / call_count
                        remaining = avg_per_call * (total_calls - call_count)
                        print(f"  진행: {call_count}/{total_calls} "
                              f"(호출당 평균 {avg_per_call:.1f}초, 남은 예상 시간 {remaining/60:.1f}분)")

                    texts = [t[1] for t in result]
                    confidences = [t[2] for t in result]
                    joined_text = " ".join(texts)
                    avg_conf = float(np.mean(confidences)) if confidences else 0.0

                    row = {
                        "image": image_path,
                        "preprocess": prep_name,
                        **d_params,
                        **r_params,
                        "predicted_text": joined_text,
                        "avg_confidence": round(avg_conf, 4),
                        "cer": round(cer(gt, joined_text), 4) if gt else "N/A",
                        "elapsed_sec": round(elapsed, 3),
                    }
                    results.append(row)

    return results


def save_results(results, out_path="results.csv"):
    if not results:
        print("결과가 없습니다.")
        return
    keys = list(results[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"{len(results)}개 조합 결과 저장 완료: {out_path}")


if __name__ == "__main__":
    results = run_grid_search()
    save_results(results)

    # 정답이 있는 경우 CER 기준 상위 5개 출력
    scored = [r for r in results if r["cer"] != "N/A"]
    if scored:
        scored.sort(key=lambda r: r["cer"])
        print("\n=== CER 기준 상위 5개 조합 ===")
        for r in scored[:5]:
            print(r)
    else:
        # 정답 없으면 평균 confidence 기준으로 정렬
        results.sort(key=lambda r: -r["avg_confidence"])
        print("\n=== confidence 기준 상위 5개 조합 ===")
        for r in results[:5]:
            print(r)