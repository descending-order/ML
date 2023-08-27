import dlib
import os
 
# Dlib 正向人脸检测器
detector = dlib.get_frontal_face_detector()

  # Dlib 人脸 landmark 特征点检测器
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 用来存储生成的单张人脸的路径
path_save = "images/faces_separated/"
 
  
   
def mkdir_for_save_images(self):
    if not os.path.isdir(path_save):
        os.mkdir(path_save)


def clear_images(self):
    img_list = os.listdir(path_save)
    for img in img_list:
        os.remove(path_save + img)
