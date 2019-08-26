import dronesim as airsim
import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import msvcrt
import time
import threading


client = airsim.VehicleClient(ip="192.168.1.100")
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

#client.takeoff()
#print("takeoff over!")

t1=time.time()
responses = client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.DepthPerspective,True,False)])
t2=time.time()
t=t2-t1
print(t)


class PID:
  def __init__(self):
    self.dest=0
    self.Kp=0.8
    self.Ki=0.002
    self.Kd=0.04
    self.preerror=0
    self.prederror=0
    self.out=0
    self.buf=[0,0,0,0,0]

  def setdest(self,dest):
    self.dest=dest

  def update(self,feedback):
    self.buf.insert(len(self.buf),feedback)
    self.buf.remove(self.buf[0])
    a = np.array(self.buf)
    b = np.mean(a,axis=0)
    print(b)
    error=self.dest-b
    #if error<0.1 and error>-0.1: error=0
    derror=error-self.preerror
    dderror=derror-self.prederror
    self.preerror=error
    self.prederror=derror

    deltu = self.Kp*derror + self.Ki*error + self.Kd*dderror
    self.out+=deltu

    if self.out>1.0:self.out=1.0
    elif self.out<0.0:self.out=0.0

    return self.out

pid = PID()

pitch=0.0
roll=0.0
throttle=0.0
throttle=0.0
yaw_rate=0.0
height=20.0

def FlyControl():
  global throttle
  while(True):
    h = client.getBarometerData().altitude
    throttle = pid.update(h)
    client.moveByAngleThrottle(pitch,roll,throttle,yaw_rate,0.01)
    #time.sleep(0.01)
    #print(throttle)

pid.setdest(height);
t = threading.Thread(target=FlyControl)
t.start()

while(True):
    key = msvcrt.getch()
    if(key==b'w'):
        pitch=-0.01
    if(key==b'x'):
        pitch=0.01
    if(key==b'a'):
        roll=-0.01
    if(key==b'd'):
        roll=0.01
    if(key==b'n'):
        yaw_rate=-1
    if(key==b'm'):
        yaw_rate=1
    if(key==b'o'):
        height+=1
        pid.setdest(height)
    if(key==b'p'):
        height-=1
        pid.setdest(height);
    if(key==b's'):
	    pitch=0.0;roll=0.0;yaw_rate=0.0;
    if(key==b'q'):
        break
    print("pitch="+str(pitch)+",roll="+str(roll)+",throttle="+str(throttle)+",yaw_rate="+str(yaw_rate))