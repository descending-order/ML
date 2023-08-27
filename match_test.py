import cv2
import matchModule as mm
imageA = cv2.imread("left.jpg")
imageA = cv2.resize(imageA,(0,0),None,0.2,0.2)
imageB = cv2.imread("right.jpg")
imageB = cv2.resize(imageB,(0,0),None,0.2,0.2)

# 把图片拼接成全景图
stitcher = mm.Stitcher()       # 调用拼接函数
(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

# 显示所有图片
cv2.imshow("Image A", imageA)
cv2.imshow("Image B", imageB)
cv2.imshow("Keypoint Matches", vis)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()