import cross2
import motion

import time
import cv2

t0=time.time()
cap=cv2.VideoCapture(1)
uav = motion.VehicleMotion()
cross2=cross2.cross2(uav,cap)
uav.connection()


motion.takeoff()
cross2.rowMove()
cross2.rowMove()
cross2.forwardC()


cap.release()
