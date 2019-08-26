import cv2
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
def getTrees(img):#获得树桩中心的坐标
    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(HSV)
    Lower = np.array([15, 7, 180])
    Upper = np.array([20, 22, 215])#柱子
    mask = cv2.inRange(HSV, Lower, Upper)
    points = cv2.bitwise_and(img, img, mask=mask)
    points_gray = cv2.cvtColor(points,cv2.COLOR_BGR2GRAY)
    # 侵蚀与稀释，去噪，因为黑白反转故侵蚀与稀释反转
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    points_gray = cv2.dilate(points_gray, kernel, iterations=1)
    points_gray = cv2.erode(points_gray, kernel, iterations=4)
    points_gray = cv2.dilate(points_gray, kernel, iterations=3)
    re,points1=cv2.threshold(points_gray,50,255,cv2.THRESH_BINARY_INV)#points1为二值图像 (points1[x][y] x不超过480 y不超过940）
    #x轴正半轴为下，y轴正半轴为右
    #cv2.imshow('1',points1)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    height = points1.shape[0]#480
    width = points1.shape[1]#640
    dot= []

    for raw in range(height):
        for col in range(width):
            if(points1[raw][col]==0):#黑色的是柱子
                #print(' x = '+str(raw)+' y = '+str(col))
                dot.append([raw,col])#黑色点的图像
    dot=np.array(dot)#shape (2872,2)
    #print(dot)
    x=dot[:,1]#横坐标 最大640
    y=dot[:,0]#纵坐标 最大480
    y=480-y
    #plt.scatter(x,y)#打印所有点的位置
    #plt.show()
    dot=np.array(dot)

    estimator = KMeans(n_clusters=15)#构造聚类器
    estimator.fit(dot)#聚类
    label_pred = estimator.labels_ #获取聚类标签
    centroids = estimator.cluster_centers_ #获取聚类中心
    inertia = estimator.inertia_ # 获取聚类准则的总和
    centroids[:,0]=480-centroids[:,0]
    #print(centroids)
    centroids=centroids[np.lexsort(centroids[:,:-1].T)]
    x_cen=centroids[:,1]
    y_cen=centroids[:,0]
    #y=480-y
    plt.scatter(x_cen,y_cen)
    plt.show()

    return centroids

img=cv2.imread('/Users/tongbohan/Desktop/images/img_3_0_1562030342490023600.png')
#print(getTrees(img))
a = getTrees(img)
a[[0, 1], :] = a[[1, 0], :] # 实现了第i行与第j行的互换
#print(a)
path = a/100
#print(path)
def flyByVector(vector):#传入一个移动向量
    x = vector[0]
    y = vector[1]
    if x>0:#如果大于0向前或者向右
        uav.flyCmd('forward','fast')
        time.sleep(x)
        #这里的sleep参数不确定，需要测试得到结果，即移动的时间和像素之间的关系
    elif x<0:
        uav.flyCmd('backward','fast')
        time.sleep(-x)
    else:uav.flyCmd('stop')

    if y>0:
        uav.flyCmd('moveright','fast')
        time.sleep(y)
        #这里的sleep参数不确定
    elif y<0:
        uav.flyCmd('moveleft','fast')
        time.sleep(-y)
    else:uav.flyCmd('stop')


for num in range(path.shape[0]):
        print('now going to tree : '+str(num+1))
        flyByVector(path[num])
