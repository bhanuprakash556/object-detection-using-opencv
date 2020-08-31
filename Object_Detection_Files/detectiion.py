# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 19:44:01 2020

@author: bhanu prakash
"""
import cv2

#img = cv2.imread("images.jpg")
#print(img.shape)

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

classNames = []
classFile = "coco.names"

with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")


configpath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightspath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightspath,configpath)
print("model imported")
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

while True:
    success , img = cap.read()
    classIds , confs ,bbox = net.detect(img,confThreshold = 0.5)
    print(classIds,bbox)
    
   
    
    
    if len(classIds)!=0:       
        for classId, confidence , box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color = (0,0,0),thickness  = 0)
            cv2.putText(img,
                        classNames[classId-1] , 
                        (box[0]+10,box[1]+30) ,
                        cv2.FONT_HERSHEY_COMPLEX_SMALL ,
                        1 ,
                        (255,0,0) ,
   
                        2
                        )
        
    if  cv2.waitKey(1) & 0xFF ==ord("q"):
            break
    cv2.imshow("image",img)
    cv2.waitKey(1)
    
    
    
    
