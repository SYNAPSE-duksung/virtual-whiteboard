import cv2
import numpy as np

def enhance_image(image_path, output_path):

    # 이미지 읽기
    img = cv2.imread(image_path)

    # 그레이스케일
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 이진화 (글자를 흰색으로 만들기 위해 INV 사용)
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # ============================
    # 글씨 굵게 만들기
    # ============================

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)          # 5,5 / 7,7 / 9,9 테스트
    )

    dilated = cv2.dilate(
        binary,
        kernel,
        iterations=3    # 2~5 정도 추천
    )

    # 다시 검은 글씨로 반전
    result = 255 - dilated

    cv2.imwrite(output_path, result)


def main():

    input_image = "./easyOCR_testImg/img_001_3px.png"
    output_image = "./easyOCR_testImg/dilate_img_001_3px.png"

    enhance_image(input_image, output_image)

    print("완료!")


if __name__ == "__main__":
    main()