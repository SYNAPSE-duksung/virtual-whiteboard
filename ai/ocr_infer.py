"""
OCR 모듈

역할:
- preprocess.py에서 전처리한 이미지를 받아 EasyOCR로 텍스트를 인식한다.
- DETECT_PARAMS / RECOGNIZE_PARAMS는 파라미터 튜닝 스크립트에서
  찾은 최적 조합으로 바꿔서 쓰면 된다. (지금은 기본값)

사용법 (단독 실행):
    python ocr_infer.py 이미지경로.jpg
"""


from pathlib import Path

import easyocr
import numpy as np

try:
    from ai.preprocess import DEFAULT_METHOD, preprocess_array, preprocess_image
except ImportError:
    from preprocess import DEFAULT_METHOD, preprocess_array, preprocess_image


LANGS = ["ko", "en"]
GPU = False

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# preprocess.py의 DEFAULT_METHOD를 그대로 따른다 (문자열을 여기서 다시 하드코딩하면
# 두 파일이 조용히 어긋날 수 있음).
PREPROCESS_METHOD = DEFAULT_METHOD

# 튜닝 결과로 나온 최적 조합으로 교체 가능
DETECT_PARAMS = {
    "text_threshold": 0.5,
    "mag_ratio": 1.0,
}

RECOGNIZE_PARAMS = {
    "contrast_ths": 0.1,
    "decoder": "greedy",
}


# Reader는 로딩이 오래 걸리므로 한 번만 생성해서 재사용한다.
_reader = None


def get_reader() -> easyocr.Reader:
    global _reader

    if _reader is None:
        print("EasyOCR Reader 로딩 중...")
        _reader = easyocr.Reader(LANGS, gpu=GPU)

    return _reader


def _read_text(img: np.ndarray) -> str:
    """전처리된 이미지에 OCR을 돌려 인식된 텍스트를 하나의 문자열로 합친다."""

    reader = get_reader()

    result = reader.readtext(
        img,
        detail=1,
        paragraph=False,
        **DETECT_PARAMS,
        **RECOGNIZE_PARAMS,
    )

    texts = [item[1] for item in result]
    return " ".join(texts)


def run_ocr(image_path: str, preprocess_method: str = PREPROCESS_METHOD) -> str:
    """
    이미지 경로를 받아 전처리 -> OCR을 수행하고,
    인식된 텍스트를 하나의 문자열로 합쳐서 반환한다.
    """

    img = preprocess_image(image_path, method=preprocess_method)
    return _read_text(img)


def run_ocr_array(image: np.ndarray, preprocess_method: str = PREPROCESS_METHOD) -> str:
    """
    이미 메모리에 있는 이미지(ndarray)를 받아 전처리 -> OCR을 수행한다.
    ``run_ocr``와 달리 디스크에 쓰고 다시 읽는 왕복 없이 바로 처리한다 — 캔버스
    스냅샷 등 이미 배열로 들고 있는 이미지를 넘길 때 사용한다.
    """

    img = preprocess_array(image, method=preprocess_method)
    return _read_text(img)


def collect_images(image_dir: str, recursive: bool = True) -> list[str]:
    """
    폴더 안의 이미지 파일 경로를 모두 수집한다.
    """

    root = Path(image_dir)

    if not root.exists():
        raise FileNotFoundError(f"이미지 폴더를 찾을 수 없습니다: {image_dir}")

    pattern = "**/*" if recursive else "*"

    paths = sorted(
        str(p)
        for p in root.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    if not paths:
        print(f"경고: {image_dir}에서 이미지를 찾지 못했습니다.")

    return paths


def run_ocr_folder(image_dir: str, recursive: bool = True) -> dict:
    """
    폴더 안의 모든 이미지에 대해 OCR을 수행하고
    {이미지경로: 인식된 텍스트} 형태의 dict를 반환한다.
    """

    image_paths = collect_images(image_dir, recursive=recursive)

    results = {}

    for i, path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] OCR 처리 중: {path}")

        try:
            results[path] = run_ocr(path)
        except Exception as e:
            print(f"  실패: {e}")
            results[path] = ""

    return results


if __name__ == "__main__":
    import sys

    # 폴더 경로를 넣으면 폴더 전체를, 이미지 경로를 넣으면 한 장만 처리한다.
    target = sys.argv[1] if len(sys.argv) > 1 else "./Img"

    if Path(target).is_dir():
        results = run_ocr_folder(target)

        print("\n=== OCR 결과 ===")
        for path, text in results.items():
            print(f"\n[{path}]")
            print(text)

    else:
        text = run_ocr(target)

        print("\nOCR 결과")
        print(text)