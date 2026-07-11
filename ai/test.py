import sys
for p in sys.path:
    print(p)

import cv2
print(cv2)
print(cv2.__file__)
print(cv2.__version__)
print(hasattr(cv2, "imread"))