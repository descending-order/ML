import array
import cv2
import dlib
import matplotlib.pyplot as plt

# 获取图片
 
image = cv2.imread('face.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# 使用特征提取器get_frontal_face_detector
detector = dlib.get_frontal_face_detector()

dets = detector(image_gray, 1)

for det in dets:
    # 将框画在原图上
    # cv2.rectangle  参数1：图片， 参数2：左上角坐标， 参数2：左上角坐标， 参数3：右下角坐标， 参数4：颜色（R,G,B）， 参数2：粗细
    image = cv2.resize(image,(0,0),None,0.2,0.2)
    my_img = cv2.rectangle(image, (det.left(), det.top()), (det.right(), det.bottom()), (0, 255, 0), 2)

#cv2.imshow("image", image)
#cv2.waitKey(0)


# 人脸检测器
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
#
for det in dets:
    shape = predictor(image, det)
    # 将关键点绘制到人脸上
    for i in range(68):
        cv2.putText(image, str(i), (shape.part(i).x, shape.part(i).y), cv2.FONT_HERSHEY_DUPLEX, 0.1, (0, 255,0 ), 1, cv2.LINE_AA)
        cv2.circle(image, (shape.part(i).x, shape.part(i).y), 1, (0, 0, 255))

cv2.imshow("rotated", image)
cv2.waitKey(0)

# 人脸对齐
image = dlib.get_face_chip(image, shape, size = 150)
cv2.imshow("68landmarks", image)
cv2.waitKey(0)


