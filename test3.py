import face_recognition
import cv2 # opencv读取图像的格式BGR（图像的格式RGB）
import numpy as np
import os
import saveModule as sm
path_read = "images/faces_for_test/"
img = cv2.imread(path_read + "face.jpg")
faces = sm.detector(img, 1)

print("人脸数 / faces in all:", len(faces), '\n')

for num, face in enumerate(faces):

    # 计算矩形框大小
    height = face.bottom() - face.top()
    width = face.right() - face.left()

    # 根据人脸大小生成空的图像
    img_blank = np.zeros((height, width, 3), np.uint8)

    for i in range(height):
        for j in range(width):
            img_blank[i][j] = img[face.top() + i][face.left() + j]

    # cv2.imshow("face_"+str(num+1), img_blank)

    # 存在本地
    #print("Save into:", sm.path_save + "img_face_" + str(num + 1) + ".jpg")
    cv2.imwrite(sm.path_save + "img_face_" + str(num + 1) + ".jpg", img_blank)


