import numpy as np
import cv2
import math
import time


#calculate angle
def angle(pt1,pt2,pt0):
    dx1 = pt1[0][0] - pt0[0][0]
    dy1 = pt1[0][1] - pt0[0][1]
    dx2 = pt2[0][0] - pt0[0][0]
    dy2 = pt2[0][1] - pt0[0][1]
    return float((dx1*dx2 + dy1*dy2))/math.sqrt(float((dx1*dx1 + dy1*dy1))*(dx2*dx2 + dy2*dy2) + 1e-10)

def mapping():
    frame = cv2.imread('Unknown.png')

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

    # show map


    # sort the points
    #for i in range(0,np.shape(center_dict)[0]-2):
    #print(np.shape(center_dict)[0])


    #center_dict[1] = np.shape(frame)[1]-center_dict[1]
    # cv2.imshow('map result',frame)
    # cv2.waitKey(5000)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #print(np.shape(frame)[0])
    center_dict[1] = np.shape(frame)[1]-center_dict[1]
    #c_dict = np.array([center_dict[1]])

    line1 = np.array([center_dict[0]])
    new_array = center_dict[1:].copy()
    print('line1: ')
    print(line1)
    print('\n')

    for i in range(0,np.shape(new_array)[0]):
        print(i)
        #cur_point = new_array[i]
        #print('cur point: '+str(cur_point))

        #if new_array[]

        new_array = np.delete(new_array,0,0)

        # print('i: '+str(i)+' before')
        print(new_array)





        print('\n')


    return center_dict


if __name__ == "__main__":

    a = mapping()
    print(a)
