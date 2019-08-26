import pyrealsense2 as rs
import numpy as np
import math
import cv2
import zbar
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import time
import os

class Task_2:

	def __init__(self,motion):
		self.pipeline = rs.pipeline()
		self.config = rs.config()
		self.motion = motion
		# self.cap = cv2.VideoCapture(0)

	def start_pipeline(self):
		# use these fuctions to get the color image and the depth image, if we could manage to install librealsense2 and pyrealsense2 (fuck GFW 🌸🐔)
		self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
		self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
		self.pipeline.start(self.config)
		print("pipeline started! ")
		self.frames = self.pipeline.wait_for_frames()

	def get_colorimage(self):

		# Wait for a coherent pair of frames: depth and color	
		        
		color_frame = self.frames.get_color_frame()
		# Convert images to numpy arrays
		        
		color_image = np.asanyarray(color_frame.get_data())

		return color_image

	def get_depthimage(self):

				
		# Wait for a coherent pair of frames: depth and color
				
		depth_frame = self.frames.get_depth_frame()
		# Convert images to numpy arrays
		depth_image = np.asanyarray(depth_frame.get_data())
		        
		depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)


		return depth_colormap





	def get_dis(self,x_pos,y_pos):
		count = 0
		dep_frame = self.frames.get_depth_frame()
		for i in range(-10,10):
			for j in range(-10.10):
				dis = dep_frame.get_distance(x_pos+i,y_pos+j)
				count = count+1
		#distance = dep_frame.get_distance(x_pos,y_pos)
		if count!=0:
			return distance/count
		else:
			print('failed to get the distance')
			return -1

	def get_scene_in_range(self,n):
		img_color = self.get_colorimage()
		dep_frame = self.frames.get_depth_frame()
        for i in range(640):
            for j in range(480):
                if depth_frame.get_distance(i,j)>n:
                    img_color[j][i]=255
        # cv2.imshow('scene in range : '+str(n),img_color)
        return img_color


	def stop_pipeline(self):
		self.pipeline.stop()
		print("pipeline stop")



	'''
	the funtion for code detection
	return the image including the code after processing, and the dictionary of
	detected code's data and postion
	'''
	def detect(self,img):

	    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	    # init a scanner
	    scanner = zbar.Scanner()
	    results = scanner.scan(gray)
	    # count = 0
	    # codes_dic: a dictionary that contains the information of codes detected
	    codes_dic = {}
	    for result in results:
	        # #print('the data '+str(count+1)+' is: ')
	        # data = str(result.data,'utf-8')
	        # print(data)
	        # #print('the position '+str(count+1)+' is: ')
	        # pos = result.position
	        # pos = np.array(pos)
	        # print(pos)
	        # pos_x = (pos[0][0]+pos[1][0]+pos[2][0]+pos[3][0])/4
	        # pos_y = (pos[0][1]+pos[1][1]+pos[2][1]+pos[3][1])/4
	        # # count = count+1
	        # img = cv2.fillPoly(img,[pos],(0,0,0))
	        # codes_dic[data] = pos
	        # print('the data '+str(count+1)+' is: ')
	        
	        # the data the code contained
	        data = str(result.data,'utf-8')
	        print(data)
	        #print('the position '+str(count+1)+' is: ')

	        # the positon of the code's four corners, and transform it into np.array
	        pos = result.position
	        pos = np.array(pos)
			print(pos)
			# the center position of the code
	        pos_x = (pos[0][0]+pos[1][0]+pos[2][0]+pos[3][0])/4
	        pos_y = (pos[0][1]+pos[1][1]+pos[2][1]+pos[3][1])/4
	        #cv2.putText(img,data,(pos_x,pos_y),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,0))
	        #cv2.putText(img,data,(int(pos_x),int(pos_y)),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255))
	        codes_dic[data] = pos
	        # count = count+1
	        
	        # draw the rectangle and data in the image
	        img = cv2.rectangle(img,(pos[0][0],pos[0][1]),(pos[2][0],pos[2][1]),(0,0,0),2)
	        cv2.putText(img,data,(int(pos_x),int(pos_y)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

	    if len(codes_dic)!=0:
	        print('find '+str(codes_dic)+' codes')
	        print(codes_dic)
	        return img,codes_dic
	    # return img,codes_dic
		else:
			return None, None




	'''
		when the UAV reaches the stump, align the UAV to the code, adjust distance
		and save image data

		if there is a code which has been detected and processed successfully, 
		return True, else return False 
	'''
	def search(self):
		
		color_image = self.get_colorimage()
    	depth_image = self.get_depthimage()
      	directory = {}
        img,dic = self.detect(color_image)
        for i,j in dic.items():
        	# i: the data   type: string
        	# j: the position of the code's four corners. type: numpy array
        		
        	# get the distance to the code
        	pos_x = (j[0][0]+j[1][0]+j[2][0]+j[3][0])/4
        	pos_y = (j[0][1]+j[1][1]+j[2][1]+j[3][1])/4
        	dis = self.get_dis(pos_x,pos_y)
        	directory[np.array(pos_x,pos_y)] = dis
        	cv2.putText(img,"dis: "+str(dis),int(pos_x),int(pos_y+25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
        	images = np.hstack((img, depth_colormap))
			# Show images
	        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
	        cv2.imshow('RealSense', images)

        if len(directory)!=0:
	        # find the max code in the image and return its position in the image
	        max_code_position = max(directory, key = directory.get)
	        print('the max code is :'+str(directory[max_code_position]))
	        target_x = max_code_position[0]
	        target_y = max_code_position[1]
	        # the error range
	        err = 5
	        print('start moving to the code')
	        # move in horizon
	        if target_x>320+err and target_y>=240-err and target_y<==240+err:
	        	self.motion.flyCmd('moveright')
	        elif target_x<320-err and target_y>==240-err and target_y<==240+err:
	        	self.motion.flyCmd('moveleft')
	        elif target_x>320+err and target_y>240+err:
	        	self.motion.flyCmd('moveright')
	        	self.motion.flyCmd('down')
	        elif target_x<320-err and target_y>240+err:
	        	self.motion.flyCmd('moveleft')
	        	self.motion.flyCmd('down')
	        elif target_x<320-err and target_y<240-err:
	        	self.motion.flyCmd('moveleft')
	        	self.motion.flyCmd('up')
	        elif target_x>320+err and target_y<=240-err:
	        	self.motion.flyCmd('moveright')
	        	self.motion.flyCmd('up')
	        elif target_x<=320+err and target_x>=320-err and target_y>240+err:
	        	self.motion.flyCmd('down')
	        elif target_x<=320+err and target_x>=320-err and target_y<240-err:
	        	self.motion.flyCmd('up')
	        elif target_x<=320+err and target_x>=320-err and target_y>=240-err and target_y<=240+err:
	        	self.motion.flyCmd('stop')
	        	time.sleep(1)
	        	print('successfully aligned')
	        	self.motion.flyCmd('stop')
	        	time.sleep(3)

	        # move forward until distance is almost 1 meter
	        while True:
	        	
	        	if self.get_dis(target_x,target_y)>1.05:
	        		self.motion.flyCmd('forward')
	        	if self.get_dis(target_x,target_y)<0.95:
	        		self.motion.flyCmd('backward')
	        	if self.get_dis(target_x,target_y)>=0.95 and self.get_dis(target_x,target_y)=<1.05:
	        		self.motion.flyCmd('stop')
	        		time.sleep(1)
	        		print('successfully reached 1 meter, stop')
	        		break

	        print('saving the image...')
	        cv2.imwrite(str(directory[max_code_position])+'.jpg',color_image)
	        print('successfully saved '+str(directory[max_code_position])+'.jpg')

	        return True

	    else:
	    	print('find no code!')
	    	return False

	
	'''
	after task 1 is finished, lift up the UAV a little, and then get the map 
	of the stump with the camera, using kmeans algorithm

	return a numpy array of probable positon of each stump

	ps: I'm not sure that if this fuction will give back the correct 
		position of the stumps, but I have no better solution so far.
		Or maybe to detect the rectangle in the img[:,240:480]
		We may need a better method to find the stump.
	'''
	
	def get_stump_map_1(self):
		
		# color_image = self.get_colorimage()
		depth_image = self.get_depthimage()
		range_image = self.get_scene_in_range(10)
		
		# get image of the second half within the distance of 10 meters
		range_image = range_image[:,240:480]
		
		HSV = cv2.cvtColor(range_image, cv2.COLOR_BGR2HSV)
    	H, S, V = cv2.split(HSV)
    	Lower = np.array([0, 0, 221])
    	Upper = np.array([180, 30, 255])#柱子
    	mask = cv2.inRange(HSV, Lower, Upper)
    	points = cv2.bitwise_and(img, img, mask=mask)
    	points_gray = cv2.cvtColor(points,cv2.COLOR_BGR2GRAY)
    	# 侵蚀与稀释，去噪，因为黑白反转故侵蚀与稀释反转
    	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    	points_gray = cv2.dilate(points_gray, kernel, iterations=1)
    	points_gray = cv2.erode(points_gray, kernel, iterations=4)
    	points_gray = cv2.dilate(points_gray, kernel, iterations=3)
    	re,points1=cv2.threshold(points_gray,50,255,cv2.THRESH_BINARY_INV)#points1为二值图像
    	#x轴正半轴为下，y轴正半轴为右
    	#cv2.imshow('1',points1)
    	#cv2.waitKey(0)
    	#cv2.destroyAllWindows()
    	height = points1.shape[0]#240
    	width = points1.shape[1]#640
    	dot= []

    	for raw in range(height):
        	for col in range(width):
            	if(points1[raw][col]==0):#黑色的是柱子
                	#print(' x = '+str(raw)+' y = '+str(col))
                	dot.append([raw,col])#黑色点的图像
    	dot=np.array(dot)#shape (the number of dots,2)
    	#print(dot)
    	x=dot[:,1]#横坐标 最大640
	    y=dot[:,0]#纵坐标 最大240
	    y=240-y
	    #plt.scatter(x,y)#打印所有点的位置
	    #plt.show()
	    dot=np.array(dot)

	    estimator = KMeans(n_clusters=10)#构造聚类器
	    estimator.fit(dot)#聚类
	    label_pred = estimator.labels_ #获取聚类标签
	    centroids = estimator.cluster_centers_ #获取聚类中心
	    inertia = estimator.inertia_ # 获取聚类准则的总和
	    centroids[:,0]=240-centroids[:,0]
	    #print(centroids)
	    # sort the dots, incremented by x_pos
	    centroids=centroids[np.lexsort(centroids[:,:-1].T)]
	    x_cen=centroids[:,1]
	    y_cen=centroids[:,0]
	    #y=240-y
	    plt.scatter(x_cen,y_cen)
	    plt.show()

	    return centroids



'''
a new version, using rectangle dectection
make the UAV up a little, and then get the position of stumps, 
as the top of the stumps are in shape of rectangle
'''

	#calculate angle
	def angle(pt1,pt2,pt0):
	    dx1 = pt1[0][0] - pt0[0][0]
	    dy1 = pt1[0][1] - pt0[0][1]
	    dx2 = pt2[0][0] - pt0[0][0]
	    dy2 = pt2[0][1] - pt0[0][1]
	    return float((dx1*dx2 + dy1*dy2))/math.sqrt(float((dx1*dx1 + dy1*dy1))*(dx2*dx2 + dy2*dy2) + 1e-10)

	def get_stump_map_2(self):
		
		self.motion.flyCmd('up')
		time.sleep(2)
		self.motion.flyCmd('stop')

		frame = self.get_colorimage()
	    #dictionary of all contours
	    contours = {}
	    #array of edges of polygon
	    approx = []

	    #grayscale
	    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	    #Canny
	    canny = cv2.Canny(frame,80,240,3)

	    #contours
	    contours, hierarchy = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	    center_dict=np.zeros(shape=(len(contours),2))
	    #print(len(contours))
	    for i in range(0,len(contours)):
	        #approximate the contour with accuracy proportional to
	        #the contour perimeter
	        approx = cv2.approxPolyDP(contours[i],cv2.arcLength(contours[i],True)*0.02,True)

	        #Skip small or non-convex objects
	        if(abs(cv2.contourArea(contours[i]))<100 or not(cv2.isContourConvex(approx))):
	            continue

	        #nb vertices of a polygonal curve
	        vtc = len(approx)
	        #get cos of all corners
	        cos = []
	        for j in range(2,vtc+1):
	            cos.append(angle(approx[j%vtc],approx[j-2],approx[j-1]))
	        #sort ascending cos
	        cos.sort()
	        #get lowest and highest
	        mincos = cos[0]
	        maxcos = cos[-1]
	        print(cos)

	        #Use the degrees obtained above and the number of vertices
	        #to determine the shape of the contour
	        # (x,y): the top left corner point's position
	        # w: width, h: height
	        x,y,w,h = cv2.boundingRect(contours[i])

	        # print(x)
	        # print(y)
	        # print(w)
	        # print(h)
	        if(vtc==4):
	            cv2.putText(frame,'RECT'+str(i+1),(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
	            #cv2.rectangle(frame,(x,y),(int(x+w/2),int(y+h/2)),(0,0,0),2)
	            cv2.circle(frame,(int(x+w/2),int(y+h/2)),1,(0,0,255),9)
	            print('rect find')
	            print(int(x+w/2),int(y+h/2))
	            print('\n')
	            center_dict[i] = (int(x+w/2),int(y+h/2))
	            #center_dict = np.reshape(center_dict,[len(contours),2])

	    if len(center_dict)!=0:
	    	print('successfully get the map')

	    # show map
	    cv2.imshow('map result',frame)
	    cv2.waitKey(5000)
	    cv2.destroyAllWindows()

	    # sort the points, left-down corner is the (0,0) point
	    center_dict[1] = np.shape(frame)[1]-center_dict[1]
	    
	    return center_dict

'''
	make the UAV fly with the map(ver.1)
'''
	def fly_with_map_ver1(self,map,count):

		# first_path = np.array([320,240])-np.array([map[0][0],map[0][1]])
		print('now going to tree 1')
		self.motion.flyCmd('down')
		time.sleep(2)
		
		while True:

			while self.search() is not True:
				a = map[count+1][0]-map[count][0]
				b = map[count+1][1]-map[count][1]
				if a<0:
					# self.motion.flyCmd('stop')
					# time.sleep(1)
					self.motion.flyCmd('moveleft')
					time.sleep(a/10)
					# a becomes a/10 for safety
					# we may need to transform the camera coordinate system
					# into ground coordinate system
					self.motion.flyCmd('stop')
				elif a>0:
					# self.motion.flyCmd('stop')
					# time.sleep(1)
					self.motion.flyCmd('moveright')
					time.sleep(a/10)
					self.motion.flyCmd('stop')
					
				elif b>1:
					# self.motion.flyCmd('stop')
					# time.sleep(1)
					self.motion.flyCmd('forward')
					time.sleep(b/10)
					self.motion.flyCmd('stop')
					# self.motion.flyCmd('stop')
					# time.sleep(0.5)
				elif b<-1:
					# self.motion.flyCmd('stop')
					# time.sleep(1)
					self.motion.flyCmd('backward')
					time.sleep(b/10)
					self.motion.flyCmd('stop')
					# self.motion.flyCmd('stop')
					# time.sleep(0.5)

'''
	give a vector, as np.array([100,200]), and then the UAV will fly on the given direction
'''

	def fly_with_vector(self,move_vector):

		a = move_vector[0]
		b = move_vector[1]
		
		
		if b>70:# this number may have to change later

			# self.motion.flyCmd('stop')
			# time.sleep(1)
			self.motion.flyCmd('forward')
			time.sleep(b/10)
			self.motion.flyCmd('stop')
			# self.motion.flyCmd('stop')
			# time.sleep(0.5)
		elif b<-70:
			# self.motion.flyCmd('stop')
			# time.sleep(1)
			self.motion.flyCmd('backward')
			time.sleep(b/10)
			self.motion.flyCmd('stop')
			# self.motion.flyCmd('stop')
			# time.sleep(0.5)

		if a<0:
			# self.motion.flyCmd('stop')
			# time.sleep(1)
			self.motion.flyCmd('moveleft')
			time.sleep(a/10)

			self.motion.flyCmd('stop')
		elif a>0:
			# self.motion.flyCmd('stop')
			# time.sleep(1)
			self.motion.flyCmd('moveright')
			time.sleep(a/10)
			self.motion.flyCmd('stop')
			



'''
	fly with map(ver.2)
	we should consider about stumps in the next line
	not finished yet
'''
	def fly_with_map_ver2(self,map,count):
			# two special condition: first and next line
			if count == 1:
				# first_path = np.array([320,240])-np.array([map[0][0],map[0][1]])
				first_path = np.array([320,0])-np.array([map[0][0],0])
				print('now going to tree 1...')
				fly_with_vector(first_path)
				self.motion.flyCmd('down')
				time.sleep(2)
				self.motion.flyCmd('stop')

			if count == 6:# or a<0 and b>10:
					# go to the next line
				print('go to the next line...')
				self.motion.flyCmd('moveright')
				time.sleep(4)
				self.motion.flyCmd('stop')
				self.motion.flyCmd('forward')
				time.sleep(1)
				self.motion.flyCmd('stop')
				self.motion.flyCmd('moveleft')
				time.sleep(4)
				self.motion.flyCmd('stop')
				print('reached next line')



		while True:

			if self.get_dis(self,320,240)>2:

				a = map[count+1][0]-map[count][0]
				b = map[count+1][1]-map[count][1]



				elif a<0:
					# self.motion.flyCmd('stop')
					# time.sleep(1)
					self.motion.flyCmd('moveleft')
					time.sleep(a/10)
					# a becomes a/10 for safety
					# we may need to transform the camera coordinate system
					# into ground coordinate system
					self.motion.flyCmd('stop')
				elif a>0:
					# self.motion.flyCmd('stop')
					# time.sleep(1)
					self.motion.flyCmd('moveright')
					time.sleep(a/10)
					self.motion.flyCmd('stop')
					
				elif b>1:
					# self.motion.flyCmd('stop')
					# time.sleep(1)
					self.motion.flyCmd('forward')
					time.sleep(b/10)
					self.motion.flyCmd('stop')
					# self.motion.flyCmd('stop')
					# time.sleep(0.5)
				elif b<-1:
					# self.motion.flyCmd('stop')
					# time.sleep(1)
					self.motion.flyCmd('backward')
					time.sleep(b/10)
					self.motion.flyCmd('stop')
					# self.motion.flyCmd('stop')
					# time.sleep(0.5)

	def fly_with_route(self):
		# we don't need to fly with such a disgusting map, try only to fly with fixed route, because the position of the stump is fixed, and they are straight
		fly_flag = np.array([1,1,-1,-1,-1,1,1,1,0])
		for i in range(0,9):
			cur_flag = fly_flag(n)
			next_flag = fly_flag(n+1)
			
			if next_flag = cur_flag:
				# if the flags are the same
				if cur_flag>0:
					print('go right')
					while self.search is not True:
						self.motion.flyCmd('moveright')
					# calculate the distance in front of the UAV, stop if the distance is less than 1 meter.
					ds = self.get_dis()
					if ds<=1 and ds>0:
						self.motion.flyCmd('stop')
						time.sleep(2)
					print('arrive check point: '+str(i)+'\n')

				if cur_flag<0:
					print('go left')
					while self.search is not True:
						self.motion.flyCmd('moveleft')
					# calculate the distance in front of the UAV, stop if the distance is less than 1 meter.
					ds = self.get_dis()
					if ds<=1 and ds>0:
						self.motion.flyCmd('stop')
						time.sleep(2)
					print('arrive check point: '+str(i)+'\n')

			if next_flag != cur_flag:
				# if the flags are not the same
				print('moving to the next line')
				if cur_flag>0:
					self.motion.flyCmd('moveright')
					time.sleep(3)
					self.motion.flyCmd('stop')
					self.motion.flyCmd('forward')
					time.sleep(3)
					self.motion.flyCmd('stop')

				if cur_flag<0:
					self.motion.flyCmd('moveleft')
					time.sleep(3)
					self.motion.flyCmd('stop')
					self.motion.flyCmd('forward')
					time.sleep(3)
					self.motion.flyCmd('stop')


						

	'''
	the main function that controls the UAV to finish task 2
	'''
	def task_main(self):
       # version 1
       #   self.get_stump_map_2()

       #   #array = [1,2,3,4,5,6,7,8,9,10]
       #   stump_conut = 1
       #   while stump_conut<11:
       #   	self.fly_with_map(stump_conut)
       #   	#time.sleep(10)
       #   	print('stump: '+str(stump_conut)+' arrived!')
       #   	stump = stump+1
       #   	#if self.search() is True:
    			# self.flyCmd('stop')
    			# time.sleep(1)
    			# continue

    	# version 2
    	# self.fly_with_route()

    	# version 3
    	self.fly_with_route()
        print('task finished!')
        print('landing...')
        self.motion.flyCmd('moveleft')
        time.sleep(2)
        self.motion.flyCmd('forward')
        time.sleep(4)
        self.motion.flyCmd('stop')
        time.sleep(0.5)
        self.motion.land()
        print('total fly time: '+str(self.motion.getFlyTime()))
        time.sleep(1)
        self.motion.disconnect()










