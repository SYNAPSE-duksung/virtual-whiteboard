import cv2
import numpy as np

def rotate_image(img, angle):
    h, w = img.shape[:2]

    center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        img,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated



    
def enhance_image(image_path, output_path):

    # 이미지 읽기
    img = cv2.imread(image_path)
    rotated = rotate_image(img, 3)


    # ============================
    # 1. 먼저 확대
    # ============================
    img = cv2.resize(
        rotated,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    # ============================
    # 2. 그레이스케일
    # ============================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ============================
    # 3. 대비 향상
    # ============================
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)

    # ============================
    # 4. 노이즈 제거
    # ============================
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # ============================
    # 5. Otsu 이진화
    # ============================
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # ============================
    # 6. 아주 약하게 얇게 만들기
    # ============================
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2, 2)
    )

    binary = cv2.erode(
        binary,
        kernel,
        iterations=1
    )

    # ============================
    # 7. 다시 반전
    # ============================
    result = 255 - binary

    cv2.imwrite(output_path, result)


def main():

    input_image = "./easyOCR_testImg/img_009_21px_68L_lying.png"
    output_image = "./enhanced_v2/erode_img_009_21px_68L_lying_v2.png"

    enhance_image(input_image, output_image)

    print("완료!")


if __name__ == "__main__":
    main()