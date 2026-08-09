import os
import glob
import cv2
import numpy as np
import pandas as pd


def analyze(image_path):

    img = cv2.imread(image_path)

    if img is None:
        return None

    img_h, img_w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Otsu 이진화
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 글자 영역 찾기
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    widths = []
    heights = []

    for c in contours:

        if cv2.contourArea(c) < 20:
            continue

        x, y, w, h = cv2.boundingRect(c)

        widths.append(w)
        heights.append(h)

    # Stroke Width
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    pixels = dist[dist > 0]

    if len(pixels) > 0:
        avg_stroke = pixels.mean() * 2
        max_stroke = pixels.max() * 2
    else:
        avg_stroke = 0
        max_stroke = 0

    return {
        "파일명": os.path.basename(image_path),
        "이미지폭(px)": img_w,
        "이미지높이(px)": img_h,
        "글자개수": len(widths),
        "평균글자폭(px)": round(np.mean(widths), 2) if widths else 0,
        "평균글자높이(px)": round(np.mean(heights), 2) if heights else 0,
        "평균획두께(px)": round(avg_stroke, 2),
        "최대획두께(px)": round(max_stroke, 2)
    }


def main():

    folder = "./enhanced_v2"

    extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp")

    image_files = []

    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(folder, ext)))

    results = []

    for file in image_files:

        print(f"분석 중 : {os.path.basename(file)}")

        result = analyze(file)

        if result is not None:
            results.append(result)

    df = pd.DataFrame(results)

    print(df)

    df.to_csv(
        "analysis.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nanalysis.csv 저장 완료!")


if __name__ == "__main__":
    main()