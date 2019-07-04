import cv2
import cv2.aruco as aruco
import numpy as np

"""
Function Name : detect_markers()
Input: img (numpy array)
Output: id(int)
Purpose: This function takes the image in form of a numpy array
         as input and detects ArUco markers in the image. For each
         ArUco marker detected in image, the ids
         is returned as output for the function
         If there is no markers, return None
"""

def detect_markers(img):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 选择aruco模块中预定义的字典来创建一个字典对象
        # 这个字典是由250个marker组成的，每个marker的大小为5*5bits
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
        #创建默认的parameter:marker检测过程中所有特定的选项,如为阈值，轮廓滤波，比拉特提取提供默认参数
        parameters = aruco.DetectorParameters_create()
        # 检测是有哪些marker。（原始图像，字典列表，）
        #检测出的图像的角的列表（按照原始顺序排列的四个角（从左上角顺时针开始）），检测出的所有maker的id列表
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters = parameters)
        cv2.imshow("1",aruco.drawDetectedMarkers(img,corners,ids))
        cv2.waitKey(0)
        return ids

img=cv2.imread('//Users//tongbohan//Desktop//2.png')

#cv2.imshow('1',img_re)
#cv2.waitKey(0)
a=detect_markers(img)
print(a[0])

#cv2.waitKey(0)

