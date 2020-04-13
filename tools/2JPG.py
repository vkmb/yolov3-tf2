import os
import sys
import cv2

for file_name in os.listdir(sys.argv[1]):
    _file_name = file_name.split('.')[0]
    img = cv2.imread(os.path.join(sys.argv[1], file_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(sys.argv[1],  _file_name)+".jpg", img)
    print(file_name)