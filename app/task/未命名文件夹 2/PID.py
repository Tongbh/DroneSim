import dronesim as airsim
import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import msvcrt
import time
import threading


client = airsim.VehicleClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

#client.takeoff()
#print("takeoff over!")



class PID:
  def __init__(self):
    self.dest=0
    self.Kp=4.3
    self.Ki=0.078
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
throttle=0.6125

yaw_rate=0.0
height=15.0

def FlyControl():
  global throttle,pitch
  while(True):
    h = client.getBarometerData().altitude
    throttle = pid.update(h)
    client.moveByAngleThrottle(pitch,roll,throttle,yaw_rate,0.01)
    print(client.getImuData())
    #time.sleep(0.01)
    #print(throttle)

pitch=-0.05
time.sleep(3)
pitch=0
#roll=0.07
#yaw_rate=-0.6
print(client.getImuData().linear_acceleration)
pid.setdest(height);
FlyControl()
'''
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

'''