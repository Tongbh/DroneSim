import cv2
import numpy as np
import time

class cross2:
    def __init__(self,motion):
        self.motion=motion
	   self.cap=cv2.VideoCapture(1)
	   self.row=640
	   self.col=480        
	#self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920.0)
	#self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080.0)

    def getC(self):
        ret,img0 = self.cap.read()
	cv2.imshow("test00",img0)
	
        HSV=cv2.cvtColor(img0,cv2.COLOR_BGR2HSV)
        H,S,V=cv2.split(HSV)
        LowerYellow=np.array([11,43,46])
        UpperYellow=np.array([25,255,255])

	img=cv2.inRange(HSV,LowerYellow,UpperYellow)
	
	
        ret, img1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV)  

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
	        
        img1 = cv2.dilate(img1, kernel, iterations=1)
        img1 = cv2.erode(img1, kernel, iterations=4)
        img1 = cv2.dilate(img1, kernel, iterations=3)

        #ret, img2 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY_INV) 
	
        img = img1.astype(np.uint8)
        img = cv2.medianBlur(img, 5)

        circles = cv2.HoughCircles(
        img, cv2.HOUGH_GRADIENT, 1, 15, param1=100, param2=30, minRadius=10, maxRadius=1000)
	#print circles

        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            if i[2] > circles[0][0][2]:
                circles[0] = i
        print 'x ='+str(circles[0][0][0])+'y ='+str(circles[0][0][1])+'r ='+str(circles[0][0][2])        
	cv2.circle(img0,(circles[0][0][0],circles[0][0][1]),circles[0][0][2],(255,255,0),5)
        cv2.imshow("test0",img0)
	
        return circles[0][0]


    def rowMove(self):
        xmean=self.row/2
        ymean=self.col/2
        a=5
        while(True):
            try:
		if cv2.waitKey(1)&0xFF==ord('q'): break
                P = self.getC()
            except AttributeError:
                print "AtAttributeError time + 1"
                continue
            else:
                if P[0]>xmean+a and P[1]>=ymean-a and P[1]<=ymean+a:
                    self.motion.flyCmd('moveright')
                elif P[0]<xmean-a and P[1]>=ymean-a and P[1]<=ymean+a:
                    self.motion.flyCmd('moveleft')
                elif P[0]>xmean+a and P[1]>ymean+a:
                    self.motion.flyCmd('moveright')
                    self.motion.flyCmd('down')
                elif P[0]<xmean-a and P[1]>ymean+a:
                    self.motion.flyCmd('moveleft')
                    self.motion.flyCmd('down')
                elif P[0]<xmean-a and P[1]<ymean-a:
                    self.motion.flyCmd('moveleft')
                    self.motion.flyCmd('up')
                elif P[0]>xmean+a and P[1]<ymean-a:
                    self.motion.flyCmd('moveright')
                    self.motion.flyCmd('up')
                elif P[0]<=xmean+a and P[0]>=xmean-a and P[1]>ymean+a:
                    self.motion.flyCmd('down')
                elif P[0]<=xmean+a and P[0]>=xmean-a and P[1]<ymean-a:
                    self.motion.flyCmd('up')
                elif P[0]<=xmean+a and P[0]>=xmean-a and P[1]>=ymean-a and P[1]<=ymean+a:
                    self.motion.flyCmd('stop')
                    time.sleep(1)
                    self.motion.flyCmd('stop')
		    time.sleep(7)                   		
		    break


    def forwardC(self):
	xmean=self.row/2
        ymean=self.col/2
        a=5
        Z=self.getC()[2]
        self.motion.flyCmd('forward')
	time.sleep(1)        
	i=0
        while True:
            try:
		if cv2.waitKey(1)&0xFF==ord('q'): break
                P = self.getC()
            except AttributeError:
                try:
		    self.motion.flyCmd('forward')                    
		    P = self.getC()
                except AttributeError:
                    print "cross----"
                    while(i<100):
                        self.motion.flyCmd('forward')
			time.sleep(0.5)
			i+=1
                    self.motion.flyCmd('stop')
                    break
                else:
		    if P[0]>xmean+a:
                    	self.motion.flyCmd('moveright',1)
                    elif P[0]<xmean-a:
                    	self.motion.flyCmd('moveleft',1)
		    elif P[1]>ymean+a:
                    	self.motion.flyCmd('down',1)
                    elif P[1]<ymean-a:
                    	self.motion.flyCmd('up',1)
		    else:
			self.motion.flyCmd('forward')
		    time.sleep(0.2)
                    continue
            else:
                if P[2]<Z-80:
                    print("cross----")
                    while(i<100):
                        self.motion.flyCmd('forward')
			time.sleep(0.5)
			i+=1
                    self.motion.flyCmd('stop')
                    break
                else:
                    Z=P[2]
                    if P[0]>xmean+a:
                    	self.motion.flyCmd('moveright',1)
                    elif P[0]<xmean-a:
                    	self.motion.flyCmd('moveleft',1)
		    elif P[1]>ymean+a:
                    	self.motion.flyCmd('down',1)
                    elif P[1]<ymean-a:
                    	self.motion.flyCmd('up',1)
		    else:
			self.motion.flyCmd('forward')
		time.sleep(0.2)

