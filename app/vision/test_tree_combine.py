# 准备运行示例：PythonClient / multirotor / hello_drone.py
import dronesim as airsim
import time
import numpy as np
import cv2.aruco as aruco
import math
import cv2
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans


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

    #return np.array(dot)#前面是竖直方向 480，后面是水平方向 640
#img=cv2.imread('/Users/tongbohan/Desktop/1/cross_pic/22.png')




def transArray(array,height):#批量转换获得树桩的中心点，到图像原点之间的距离

    cx=319.5
    cy=239.5
    fx=269.5
    fy=269.5
    real=np.array([])
    for i in range(array.shape[0]):
        real_pos=np.array([0.00,0.00])

        real_pos[0]=(array[i][0]-cx)*height/fx
        real_pos[1]=(array[i][1]-cy)*height/fy
        real_pos=np.around(real_pos,decimals=2)
        #print(real_pos)
        real=np.append(real,[[real_pos[0],real_pos[1]]])

    #print(array.shape[0])
    real=np.reshape(real,(array.shape[0],array.shape[1]))
    return np.around(real,decimals=2)


#np.set_printoptions(threshold=1e6)
#arr=getTrees(img)#所有的坐标 已经排序好
#print(arr)
#arr_p=arr-arr[0]#相对第一个的位置坐标
#print(arr_p)
#print(arr.shape[0])
def detect_markers(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 选择aruco模块中预定义的字典来创建一个字典对象
    # 这个字典是由250个marker组成的，每个marker的大小为5*5bits
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
      #创建默认的parameter:marker检测过程中所有特定的选项,如为阈值，轮廓滤波，比拉特提取提供默认参数
    parameters = aruco.DetectorParameters_create()        # 检测是有哪些marker。（原始图像，字典列表，）
    #检测出的图像的角的列表（按照原始顺序排列的四个角（从左上角顺时针开始）），检测出的所有maker的id列表
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters = parameters)
    cv2.imshow("1",aruco.drawDetectedMarkers(img,corners,ids))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return ids


#开始飞行 现在假设飞机在正中央
def getPath():#获得路径

    map_original = image.getVerticalSence()
    map = getTrees(map_original)#获得所有点的地图

    path=np.array([])
    for i in range(map.shape[0]-1):
        path_i=np.array([0.00,0.00])
        path_i[0]=map[i+1][0]-map[i][0]
        path_i[1]=map[i+1][1]-map[i][1]
        #print(path_i)
        path=np.append(path,[[path_i[0],path_i[1]]])
    path=np.reshape(path,(map.shape[0]-1,map.shape[1]))

    return path

def flyByVector(vector):#传入一个移动向量
    x = vector[0]
    y = vector[1]
    if x>0:#如果大于0向前或者向右
        uav.flyCmd('forward','slow')
        time.sleep(x/4)
        #这里的sleep参数不确定，需要测试得到结果，即移动的时间和像素之间的关系
    elif x<0:
        uav.flyCmd('backward','slow')
        time.sleep(-x/4)
    else:uav.flyCmd('stop')

    if y>0:
        uav.flyCmd('moveright','slow')
        time.sleep(y)
        #这里的sleep参数不确定
    elif y<0:
        uav.flyCmd('moveleft','slow')
        time.sleep(-y)
    else:uav.flyCmd('stop')





def getLandArea(img):
    uav.flyToHeight(35)
    uav.flyCmd('forward','fast')
    time.sleep(3)

    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(HSV)
    Lower = np.array([22, 2, 192])
    Upper = np.array([30, 6, 196])#柱子
    mask = cv2.inRange(HSV, Lower, Upper)
    points = cv2.bitwise_and(img, img, mask=mask)
    points_gray = cv2.cvtColor(points,cv2.COLOR_BGR2GRAY)
    # 侵蚀与稀释，去噪，因为黑白反转故侵蚀与稀释反转
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    #points_gray = cv2.dilate(points_gray, kernel, iterations=1)
    #points_gray = cv2.erode(points_gray, kernel, iterations=4)
    #points_gray = cv2.dilate(points_gray, kernel, iterations=3)
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
    plt.scatter(x,y)#打印所有点的位置
    plt.show()
    dot=np.array(dot)

    estimator = KMeans(n_clusters=1)#构造聚类器
    estimator.fit(dot)#聚类
    label_pred = estimator.labels_ #获取聚类标签
    centroids = estimator.cluster_centers_ #获取聚类中心
    inertia = estimator.inertia_ # 获取聚类准则的总和
    centroids[:,0]=480-centroids[:,0]
    #print(centroids)
    centroids=centroids[np.lexsort(centroids[:,:-1].T)]
    x_land=centroids[:,1]
    #print(x_land)
    y_land=centroids[:,0]
    #y=480-y
    #print(y_land)
    #plt.scatter(x_land,y_land)
    #plt.show()


    #print(land)
    distance = transArray(centroids,height)
    #print(distance)
    print('find land place at : '+str(x_land)+' '+str(y_land))
    return distance



def flyRoute():
    uav.flyToHeight(35)#飞到足够高的地方
    time.sleep(4)
    img = image.getVerticalImage()
    land = getLandArea(img)
    land_path = np.array([320,240])-np.array([land[0][0],land[0][1]])
    print('now going to land area')
    flyByVector(land_path)
    time.sleep(3)
    uav.flyCmd('backward','fast')
    time.sleep(3)
    route = getPath()
    first_path = np.array([320,240])-np.array([route[0][0],route[0][1]])#第一个点
    print('now going to tree 1')
    flyByVector(first_path)#到第一个点处
    arucoDirectory = np.array([])
    for num in range(route.shape[0]):
        print('now going to tree : '+str(num+1))
        flyByVector(route[num])
        uav.flyCmd('backward','slow')
        time.sleep(1)
        uav.flyCmd('down','slow')
        time.sleep(1)

        tree_ahead = uav.image.getFrontScene()
        #tree_ahead = tree_ahead[:,200:400]#裁剪
        detectedRes = detect_markers(tree_ahead)#检测到的结果
        if detectedRes is not None:
            arucoDirectory =np.append(arucoDirectory,detectedRes)
    uav.flyCmd('stop')
    last_path = np.array([3,3])
    flyByVector(last_path)
    time.sleep(4)
    uav.flyCmd('stop')

    return arucoDirectory


t1=time.time()
# 连接到AirSim模拟器
client = airsim.VehicleClient()
client.connection()

uav=airsim.VehicleMotion(client)
uav.start()

image=airsim.VehicleImage(client)
number=airsim.Number(image)


# 定位地面高度
takeoffHigh = client.getBarometerData().altitude
print('takeoffHigh =', round(takeoffHigh, 4))

cross=airsim.Cross(client,image,number,uav)

# 起飞
#client.takeoff()
uav.flyCmd('up','fast')
time.sleep(5)
uav.flyCmd('stop')
time.sleep(3)

# 定位初始方向
x0 = client.getMagnetometerData().magnetic_field_body.x_val
y0 = client.getMagnetometerData().magnetic_field_body.y_val

num=1
#1
cross.adjustPositionCentripetally()
cross.moveToCircle()
cross.adjustDrone()
cross.saveFrontSense(num)
num+=1

while(num<=10):
    cross.moveCircle_N()
    cross.adjustDrone()
    cross.saveFrontSense(num)
    num+=1

t2=time.time()
print('10-Circle complete time:',(t2-t1)/60,'min')


flyRoute()
