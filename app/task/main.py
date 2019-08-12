import cross2
import motion
import QR2_search

import time
import cv2

t0=time.time()

uav = motion.VehicleMotion()
uav.connection()
time.sleep(3)
uav.takeoff()
time.sleep(1)
uav.takeoff()
time.sleep(5)
cross2=cross2.cross2(uav)




cross2.rowMove()
cross2.rowMove()
cross2.forwardC()






# mission 2 : fast searching


# realsense camera init
realsense = QR2_search.Realsense()
realsense.start_pipeline()

# get color and depth image
while True:
	
