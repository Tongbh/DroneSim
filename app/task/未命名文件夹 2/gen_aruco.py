import cv2
import cv2.aruco as aruco
import numpy as np
def gen_aruco(n):
    while(n):

        img = np.zeros((100,100,3), np.uint8)
        Dictionary = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
        img=aruco.drawMarker(Dictionary,n,100,img,1)
        #cv2.imwrite('C:\\Users\\ailab\\PycharmProjects\\drone\\train\\'+str(n)+'.jpg',img)
        cv2.imwrite('C:\\Users\\ailab\\PycharmProjects\\drone\\train\\class_0\\' + 'aruco_0_'+str(n) + '.jpg', img)
        print('gen_aruco:  '+'aruco_0_'+str(n) + '.jpg')
        n = n - 1
    return 0

gen_aruco(999)