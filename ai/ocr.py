import easyocr
import cv2
reader = easyocr.Reader(['ko', 'en'])

result = reader.readtext("../doc/easyOCR_testImg/img_010_36px.png")

for r in result:
    print("인식 결과 :", r[1])
    print("신뢰도 :", r[2])