import sys
import time

import socket

class VehicleMotion():

    def __init__(self):
        self.s = socket.socket()
        self.init_t = time.time()
		
    def connection(self):
        self.s.connect(("192.168.66.66", 6666))
        print "m100 connected!"

    def disconnect(self):
        self.s.close()
        print "m100 disconnect!"

    def getFlyTime(self):
        now_t=time.time()
        return round(now_t-self.init_t,2)
    
	
    def takeoff(self):
        print str(self.getFlyTime())+"s: taking off..."
        self.s.send('n'.encode('utf-8'))

	
    def land(self):
        print str(self.getFlyTime())+"s: landing..."
        self.s.send('m'.encode('utf-8'))

    
    def flyCmd(self, cmd, flag=0):
        if cmd=="forward":
            self.s.send('w'.encode('utf-8'))
            print "move forward"
        elif cmd=="backward":
            self.s.send('s'.encode('utf-8'))
            print "move backward"
        elif cmd=="moveleft":
            if flag==0: self.s.send('a'.encode('utf-8'))
            elif flag==1: self.s.send('A'.encode('utf-8'))
            print "move left flag="+str(flag)
        elif cmd=="moveright":
            if flag==0: self.s.send('d'.encode('utf-8'))
            elif flag==1: self.s.send('D'.encode('utf-8'))
            print "move right flag="+str(flag)
        elif cmd=="turnleft":
            self.s.send('z'.encode('utf-8'))
            print "turn left"
        elif cmd=="turnright":
            self.s.send('x'.encode('utf-8'))
            print "turn right"
        elif cmd=="up":
            if flag==0: self.s.send('c'.encode('utf-8'))
            elif flag==1: self.s.send('C'.encode('utf-8'))
            print "move up flag="+str(flag)
        elif cmd=="down":
            if flag==0: self.s.send('v'.encode('utf-8'))
            elif flag==1: self.s.send('V'.encode('utf-8'))
            print "move down flag="+str(flag)
        elif cmd=="stop":
            self.s.send('p'.encode('utf-8'))
            print "stop"
        elif cmd=="exit":
            self.s.send('q'.encode('utf-8'))
            print "exit"

