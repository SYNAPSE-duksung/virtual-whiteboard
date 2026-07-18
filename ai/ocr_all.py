import os
import glob
import easyocr

# OCR 모델 로드 (한 번만)
reader = easyocr.Reader(['ko', 'en'])

input_dir = "../doc/easyOCR_testImg"

# 이미지 목록 가져오기
image_paths = []
image_paths.extend(glob.glob(os.path.join(input_dir, "*.png")))
image_paths.extend(glob.glob(os.path.join(input_dir, "*.jpg")))
image_paths.extend(glob.glob(os.path.join(input_dir, "*.jpeg")))

print(f"총 {len(image_paths)}개의 이미지를 찾았습니다.\n")

for image_path in sorted(image_paths):

    filename = os.path.basename(image_path)

    print("=" * 60)
    print(f"파일 : {filename}")

    result = reader.readtext(image_path)

    if len(result) == 0:
        print("인식된 글자가 없습니다.")
        continue

    for i, r in enumerate(result, start=1):
        print(f"[{i}]")
        print("인식 결과 :", r[1])
        print("신뢰도   :", f"{r[2]:.4f}")

print("\n모든 OCR이 완료되었습니다.")