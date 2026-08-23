"""
전처리 모듈

역할:
- 이미지를 읽어서 OCR에 넣기 좋은 형태로 전처리한다.
- 기존 파라미터 튜닝 스크립트에서 쓰던
  gray / clahe / adaptive_thresh 세 가지 방식을 함수로 분리했다.

사용법 (단독 실행):
    python preprocess.py 이미지경로.jpg
    python preprocess.py 이미지경로.jpg adaptive_thresh
"""


import cv2
import numpy as np


def preprocess_image(image_path: str, method: str = "clahe") -> np.ndarray:
    """
    이미지를 읽어서 지정한 방식으로 전처리한 결과를 반환한다.

    method:
        - "gray"            : 그레이스케일만 적용 (기준값)
        - "clahe"           : 대비 개선 (기본값)
        - "adaptive_thresh" : 배경 밝기가 일정하지 않을 때 사용
    """

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if method == "gray":
        return gray

    elif method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    elif method == "adaptive_thresh":
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            15,
        )

    else:
        raise ValueError(f"알 수 없는 전처리 method: {method}")


if __name__ == "__main__":
    import sys

    image_path = sys.argv[1] if len(sys.argv) > 1 else "./sample.jpg"
    method = sys.argv[2] if len(sys.argv) > 2 else "clahe"

    result = preprocess_image(image_path, method=method)

    out_path = f"preprocessed_{method}.png"
    cv2.imwrite(out_path, result)

    print(f"전처리 완료 ({method}): {out_path}")
