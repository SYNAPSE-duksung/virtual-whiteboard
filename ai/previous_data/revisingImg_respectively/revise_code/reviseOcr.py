# 글씨를 보정하는 코드
import cv2
import numpy as np
from PIL import Image, ImageEnhance


def crop_text_region(binary_img, original_img):
    """
    글씨가 있는 영역만 crop
    """

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        morph,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # 너무 작은 노이즈 제거
        if w * h > 200:
            boxes.append((x, y, w, h))

    if len(boxes) == 0:
        return original_img

    xs = [b[0] for b in boxes]
    ys = [b[1] for b in boxes]
    xe = [b[0] + b[2] for b in boxes]
    ye = [b[1] + b[3] for b in boxes]

    margin = 10

    x1 = max(min(xs) - margin, 0)
    y1 = max(min(ys) - margin, 0)
    x2 = min(max(xe) + margin, original_img.shape[1])
    y2 = min(max(ye) + margin, original_img.shape[0])

    crop = original_img[y1:y2, x1:x2]

    return crop


def enhance_image(image_path, output_path):

    #########################
    # PIL 전처리
    #########################

    image = Image.open(image_path)

    image = ImageEnhance.Brightness(image).enhance(1.2)
    image = ImageEnhance.Contrast(image).enhance(1.8)
    image = ImageEnhance.Sharpness(image).enhance(2.5)

    image_np = np.array(image)

    #########################
    # OpenCV 전처리
    #########################

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # 노이즈 제거
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # 대비 향상
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)

    #########################
    # Adaptive Threshold
    #########################

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        15
    )

    # 검은 글씨 -> 흰 글씨
    binary = 255 - binary

    #########################
    # 글씨 영역 Crop
    #########################

    crop = crop_text_region(binary, image_np)

    #########################
    # OCR 잘 되도록 확대
    #########################

    scale = 3

    crop = cv2.resize(
        crop,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    #########################
    # Crop 후 다시 전처리
    #########################

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    gray = clahe.apply(gray)

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    #########################
    # Sharpen
    #########################

    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    binary = cv2.filter2D(binary, -1, kernel)

    #########################
    # 저장
    #########################

    cv2.imwrite(output_path, binary)


import os
import glob

def main():

    input_dir = "./easyOCR_testImg"
    output_dir = "./enhanced"

    os.makedirs(output_dir, exist_ok=True)

    # png, jpg, jpeg 모두 읽기
    image_paths = []
    image_paths.extend(glob.glob(os.path.join(input_dir, "*.png")))
    image_paths.extend(glob.glob(os.path.join(input_dir, "*.jpg")))
    image_paths.extend(glob.glob(os.path.join(input_dir, "*.jpeg")))

    print(f"{len(image_paths)}개의 이미지를 찾았습니다.")

    for image_path in image_paths:

        filename = os.path.basename(image_path)

        output_path = os.path.join(
            output_dir,
            f"enhanced_{filename}"
        )

        enhance_image(image_path, output_path)

        print(f"완료 : {filename}")

    print("모든 이미지 처리가 완료되었습니다.")


if __name__ == "__main__":
    main()