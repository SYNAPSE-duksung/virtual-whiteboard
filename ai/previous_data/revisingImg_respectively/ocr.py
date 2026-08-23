import easyocr
import cv2
reader = easyocr.Reader(['ko', 'en'])

# result = reader.readtext("./enhanced_v2/dilate_img_001_3px.png")
# result = reader.readtext("./enhanced_v2/dilate_img_003_16px.png")
# result = reader.readtext("./enhanced_v2/erode_img_004_100px.png")
# result = reader.readtext("./enhanced_v2/erode_img_005_50px.png")
# result = reader.readtext("./enhanced_v2/dilate_img_006_24px.png")
# result = reader.readtext("./enhanced_v2/erode_img_008_21px_69L.png")
result = reader.readtext("./enhanced_v2/erode_img_009_21px_68L_lying_v2.png")

for r in result:
    print("인식 결과 :", r[1])
    print("신뢰도 :", r[2])