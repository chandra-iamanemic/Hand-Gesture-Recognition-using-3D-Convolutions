# -*- coding: utf-8 -*-
"""
Created on Mon May 20 17:04:48 2019

@author: cks
"""






import cv2
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv3D, MaxPooling3D

from keras.models import load_model

#model = load_model("C:/Users/cks/Documents/practice codes/gestures_3d_cnn_models/model_6_class_100x100_attempt1.h5")
model = load_model("C:/Users/cks/Documents/practice codes/gestures_3d_cnn_models/right_up_drum_shake_attempt1.h5")


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 1)
count = 0

img_size = 28
frames = 20

live_stack = np.zeros((1,frames,img_size,img_size,3),dtype = np.uint8)

motion_list = ["No Gesture","Swipe Right ","Swipe Up ","Drumming Fingers","Shaking Hand","Other"]
if not cap.isOpened():
    print("Could not open webcam")
    exit()

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    
    
    frame_reshaped = cv2.resize(frame, (img_size,img_size))
    
    if(count<frames):
        live_stack[0,count,:,:,:] = frame_reshaped
        
    else:
        temp_stack = np.zeros((1,frames,img_size,img_size,3),dtype = np.uint8)
        
        for i in range(frames-1):
            temp_stack[0,i,:,:,:] = live_stack[0,i+1,:,:,:]
        
        temp_stack[0,frames-1,:,:,:] = frame_reshaped
            
        live_stack = temp_stack
        
    cv2.imshow('frame',frame)
    
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print (cv2.waitKey(1))
        break
        
    gesture_pred = model.predict(live_stack)
    gesture_pred_classes = np.argmax(gesture_pred,axis=1)
    gesture_pred_classes = gesture_pred_classes.reshape(-1,1)
    
    if(max(gesture_pred[0]) > 0.8):
        print(motion_list[gesture_pred_classes[0][0]])
    
    count = count+1
    
    time.sleep(0.1)
cap.release()
cv2.destroyAllWindows()



