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

from preprocess import preprocess_image


LANGS = ["ko", "en"]
GPU = False

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# preprocess.py의 method와 반드시 맞춰야 함
PREPROCESS_METHOD = "clahe"

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


def run_ocr(image_path: str, preprocess_method: str = PREPROCESS_METHOD) -> str:
    """
    이미지 경로를 받아 전처리 -> OCR을 수행하고,
    인식된 텍스트를 하나의 문자열로 합쳐서 반환한다.
    """

    img = preprocess_image(image_path, method=preprocess_method)

    reader = get_reader()

    result = reader.readtext(
        img,
        detail=1,
        paragraph=False,
        **DETECT_PARAMS,
        **RECOGNIZE_PARAMS,
    )

    texts = [item[1] for item in result]
    joined_text = " ".join(texts)

    return joined_text


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