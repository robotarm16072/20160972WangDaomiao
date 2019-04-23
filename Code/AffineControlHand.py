#!/usr/bin/env python

'''
Camshift tracker
================

This is a demo that shows mean-shift based tracking
You select a color objects such as your face and it tracks it.
This reads from video camera (0 by default, or the camera number the user enters)

http://www.robinhewitt.'F'/research/track/camshift.html

Usage:
------
    camshift.py [<video source>]

    To initialize tracking, select the object with mouse

Keys:
-----
    ESC   - exit
    b     - toggle back-projected probability visualization
'''

# Python 2/3 'F'patibility
from __future__ import print_function
import serial
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

import numpy as np
import cv2 as cv

# local module
import video
from video import presets



class SendData():
    def __init__(self, portx = "COM9", bps = 9600, timex = 0):
        self.ser = serial.Serial(portx, bps, timeout=timex)

    def Move(self, x, y, z = 0):

        data = 'F'+ str(x) + '/' + str(y) + 'L'
        print(data)
        self.ser.write(data.encode("utf-8"))


SendAPP = SendData(portx = "COM9", bps = 9600, timex = 0)





drawing = False # 鼠标左键是不是按下
mode = False # True,激活绘制正方形. False 激活绘制曲线；使用m键切换
ix,iy = -1,-1
img = 0 
keypt = np.zeros((4,2));
keyptp = np.float32([[0,0],[300,0],[300,300],[0,300]])




# 以下为进行相机坐标与真实左边的对应




# mouse callback function
reselect = True
perspflag = False
j=-1;
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode,j,reselect,keypt,perspflag
 
    if event == cv.EVENT_LBUTTONDOWN:
        j+=1
        if j==4:
            j=0
            reselect=False
            keypt = np.zeros((4,2));
        if j==0:
            reselect = True
        drawing = True
        ix,iy = x,y
        keypt[j] =x,y
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
            else:
                cv.circle(img,(x,y),5,(0,0,255),-1)
 
    elif event == cv.EVENT_LBUTTONUP:
        if j ==3:
            perspflag=True
        drawing = False
        if mode == True:
            cv.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
        else:
            cv.circle(img,(x,y),5,(0,0,255),-1)

def sendToHand(event,x,y,flags,param):
    global ix,iy,drawing,mode,j,reselect,keypt,perspflag
 
    if event == cv.EVENT_LBUTTONDOWN:
        print(x,y)
        Arm7Numpy = np.float32([[148,285.0],[-10.0,234.0],[135,185]])
        CameraNumpy = np.float32([[0,0],[300,200],[0,200]])
        M = cv.getAffineTransform(CameraNumpy,Arm7Numpy)
        MMatrx = np.matrix(M)
        Mend = np.vstack((MMatrx, np.matrix('0 0 1')))


        string = '%d;%d;1' % (x,y)
        MXY = np.matrix(string)
        HandXY = Mend*MXY

        print(HandXY)
        
        xx = HandXY[0,0]
        yy = HandXY[1,0]
        
        # xx = int(xx)
        # yy = int(yy)

        SendAPP.Move(x = xx, y = yy)

    # elif event == cv.EVENT_MOUSEMOVE:
    #     print("mouse move")
        

cv.namedWindow('image')
cap=cv.VideoCapture(1)
ret ,img = cap.read()
cv.setMouseCallback('image',draw_circle)
while(1):
    if reselect == True:
        for i in range(4):
            cv.circle(img,(int(keypt[i][0]),int(keypt[i][1])),5,(0,0,255),-1)
    if perspflag == True:
        M = cv.getPerspectiveTransform(np.float32(keypt),keyptp)
        dst = cv.warpPerspective(img,M,(300,300))
        cv.imshow('haha',dst)
        #reselect == True
    cv.imshow('image',img)
    k = cv.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = not mode
    elif k == 27:
        break
    ret ,img = cap.read()


cap.release() 
cv.destroyAllWindows()


# 指哪打哪 

cv.namedWindow('sendToHand')
cap=cv.VideoCapture(1)
ret ,img = cap.read()
cv.setMouseCallback('sendToHand',sendToHand)
while(1):

    M = cv.getPerspectiveTransform(np.float32(keypt),keyptp)
    dst = cv.warpPerspective(img,M,(300,300))
    cv.imshow('sendToHand',dst)
        #reselect == True
    k = cv.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = not mode
    elif k == 27:
        break
    ret ,img = cap.read()


cap.release() 
cv.destroyAllWindows()















class App(object):
    def __init__(self, video_src):
        self.cam = video.create_capture(video_src, presets['cube'])
        _ret, self.frame = self.cam.read()
        cv.namedWindow('HandControl')
        cv.setMouseCallback('HandControl', self.onmouse)

        self.selection = None
        self.drag_start = None
        self.show_backproj = False
        self.track_window = None

    def onmouse(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.track_window = None
        if self.drag_start:
            xmin = min(x, self.drag_start[0])
            ymin = min(y, self.drag_start[1])
            xmax = max(x, self.drag_start[0])
            ymax = max(y, self.drag_start[1])
            self.selection = (xmin, ymin, xmax, ymax)
        if event == cv.EVENT_LBUTTONUP:
            self.drag_start = None
            self.track_window = (xmin, ymin, xmax - xmin, ymax - ymin)

    def show_hist(self):
        bin_count = self.hist.shape[0]
        bin_w = 24
        img = np.zeros((256, bin_count*bin_w, 3), np.uint8)
        for i in xrange(bin_count):
            h = int(self.hist[i])
            cv.rectangle(img, (i*bin_w+2, 255), ((i+1)*bin_w-2, 255-h), (int(180.0*i/bin_count), 255, 255), -1)
        img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
        cv.imshow('hist', img)

    def run(self):

        print("serial port parameters : ", ser)
        print("Port:", ser.port)
        print("Baudrate:", ser.baudrate)
        print("ArmGuider Initialization Done.")
        flagSend = 1

        imgname_0 = 0

        while True:
            # _ret, self.frame = self.cam.read()
            _ret, originName = self.cam.read()

            M = cv.getPerspectiveTransform(np.float32(keypt),keyptp)
            dst = cv.warpPerspective(originName,M,(300,300))


            self.frame = dst


            vis = self.frame.copy()
            hsv = cv.cvtColor(self.frame, cv.COLOR_BGR2HSV)
            mask = cv.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

            if self.selection:
                x0, y0, x1, y1 = self.selection
                hsv_roi = hsv[y0:y1, x0:x1]
                mask_roi = mask[y0:y1, x0:x1]
                hist = cv.calcHist( [hsv_roi], [0], mask_roi, [16], [0, 180] )
                cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX)
                self.hist = hist.reshape(-1)
                self.show_hist()

                vis_roi = vis[y0:y1, x0:x1]
                cv.bitwise_not(vis_roi, vis_roi)
                vis[mask == 0] = 0

            if self.track_window and self.track_window[2] > 0 and self.track_window[3] > 0:
                self.selection = None
                prob = cv.calcBackProject([hsv], [0], self.hist, [0, 180], 1)
                prob &= mask
                term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )
                track_box, self.track_window = cv.CamShift(prob, self.track_window, term_crit)

                if self.show_backproj:
                    vis[:] = prob[...,np.newaxis]
                try:
                    cv.ellipse(vis, track_box, (0, 0, 255), 2)


                    center = (int(track_box[0][0]),int(track_box[0][1]))
                    # print(center)



                    cv.circle(vis, center, 1, (0,0,255), 3)
                    # print(center)
               
                    if(ser.read()== b'1'):
                        flagSend = 1
                        print(1)
                    
                    data = 'F'+ str((center[0])) + '/' + str((center[1])) + 'L'
                    if flagSend == 1:
                        if(center[0]!= 0):
                            if (center[1]!=0):
                                flagSend = 0
                                print(data)
                                ser.write(data.encode("utf-8"))
                                



                    # print(track_box)
                except:
                    print(track_box)
                    x = 1

            cv.imshow('HandControl', vis)

            ch = cv.waitKey(5)
            if ch == 27:
                break
            if ch == ord('b'):
                self.show_backproj = not self.show_backproj
            if ch == ord('s'):
                cv.imwrite("./data/%d.png" % imgname_0,vis)
                imgname_0  = imgname_0 + 1
                print("%d SaveSuccessful" % imgname_0)



            

        
        cv.destroyAllWindows()







if __name__ == '__main__':
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    video_src = 1 
    print(__doc__)
    App(video_src).run()
