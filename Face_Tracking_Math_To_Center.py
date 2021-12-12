import cv2
import mediapipe as mp
import time
import numpy as np
import serial


import math
wCam, hCam=1280,720
Move_x_previous=0
Move_x=0
#https://www.youtube.com/watch?v=jn1HSXVmIrA&t=2073s
class FaceDetector():
    def __init__(self,minDetectionCon=0.7):
        self.minDetectionCon=minDetectionCon
        self.mpFaceDetection=mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(minDetectionCon)
    def findFaces(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # print(id,detection)
                # print(detection.score)

                #print(detection.location_data.relative_bounding_box)

                bboxC = detection.location_data.relative_bounding_box

                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                bboxs.append([id,bbox,detection.score])
                if draw:
                    img = self.fancyDraw(img,bbox)
                    cv2.rectangle(img, bbox, (255, 0, 255), 2)
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20),
                                cv2.FONT_HERSHEY_PLAIN,
                                3, (0, 255, 0), 2)
                    center_X, center_Y = bbox[0] + (bbox[2] / 2), bbox[1] + (bbox[3]/2)
                    length = math.hypot(wCam/2-center_X, hCam/2 + center_Y)
                    #cv2.line(img, (int(wCam/2), int(hCam/2)), (int(center_X), int(center_Y)), (255, 0, 255), 2)
                    # Hand Range from 250 to 30
                    # Volume (-64.0, 0.0, 0.03125)
                    Move_x=wCam/2-center_X
                    Move_y=hCam/2-center_Y
                    #print('X:'+ str(Move_x))
                    #print('Y:'+str(Move_y))
                    #print(int(length))




        return img, bboxs, Move_x, Move_y

    def fancyDraw(self,img,bbox,l=30,t=10,rt=1):
        x,y,w,h =bbox
        x1,y1 =x+w,y+h
        #Top Left
        cv2.line(img,(x,y),(x+l,y),(0,0,255),t)
        cv2.line(img,(x,y),(x,y+l),(0,0,255),t)
        #Top Right
        cv2.line(img, (x1, y), (x1 - l, y), (0,0,255), t)
        cv2.line(img, (x1, y), (x1, y + l), (0,0,255), t)
        #Bottom Left
        cv2.line(img, (x, y1), (x + l, y1), (0,0,255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (0,0,255), t)
        #Bottom Right
        cv2.line(img, (x1, y1), (x1 - l, y1), (0,0,255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (0,0,255), t)

        # Left Target Line
        bbox_center_Y = int(bbox[1] + bbox[3] / 2)
        Y_Target = bbox_center_Y
        bbox_center_X = int(bbox[1])
        X_Target = bbox_center_X
        cv2.line(img, (0, Y_Target), (bbox[0], Y_Target), (255, 0, 255), 2)

        # Right Target Line
        # Right Line Y
        Right_Line_Y = int(bbox[2] / 2)
        cv2.line(img, (wCam, Y_Target), (bbox[0] + bbox[2], bbox[1] + Right_Line_Y), (255, 0, 255), 2)
        # mpDraw.draw_detection(img,detection)
        # Top Target Line
        cv2.line(img, (bbox[0] + int(bbox[2] / 2), 0), (bbox[0] + int(bbox[2] / 2), bbox[1]), (255, 0, 255), 2)

        # Bottom Target Line
        cv2.line(img, (bbox[0] + int(bbox[2] / 2), wCam), (bbox[0] + int(bbox[2] / 2), (bbox[1] + bbox[3])),(255, 0, 255), 2)
        return img

def main():
    Move_x_previous=0
    Move_yy=0
    Move_xx=0
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)
    pTime = 0
    detector= FaceDetector()
    Arduino = serial.Serial('COM5', 9600, timeout=1)
    while True:
        success, img = cap.read()
        img,bboxs,Move_xx,Move_yy=detector.findFaces(img)
        #print(bboxs)
        try:
            if ((Move_x_previous/Move_xx)*100 >5):
                msg=(int(Move_xx))
                Arduino.write(b'L'))
                print(msg)
            else:
                Move_x_previous=Move_xx
        except:
            continue

        cv2.imshow("Image",img)
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime

        cv2.putText(img,f'FPS: {int(fps)}', (20,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__=="__main__":
    main()
