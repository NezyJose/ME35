# Core opencv code provided by Einsteinium Studios

# Revisions to work with Pi Camera v3 by Briana Bouchard

import numpy as np
import cv2
from picamera2 import Picamera2
from libcamera import controls
import RPi.GPIO as GPIO
import time

picam2 = Picamera2() # assigns camera variable
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous}) # sets auto focus mode
picam2.start() # activates camera
time.sleep(5) # wait to give camera time to start up

#motor pins
OUT1 = 12 #backward left
OUT2 = 11 #forward left
OUT3 = 13 #backward right
OUT4 = 15 #forward right
# Set the GPIO pins as output
GPIO.setmode(GPIO.BOARD)
GPIO.setup(OUT1, GPIO.OUT)
GPIO.setup(OUT2, GPIO.OUT)
GPIO.setup(OUT3, GPIO.OUT)
GPIO.setup(OUT4, GPIO.OUT)
GPIO.output(OUT1,GPIO.LOW)        
GPIO.output(OUT2,GPIO.LOW)
GPIO.output(OUT3,GPIO.LOW)        
GPIO.output(OUT4,GPIO.LOW)
p1=GPIO.PWM(OUT1,50)
p2=GPIO.PWM(OUT2,50)
p3=GPIO.PWM(OUT3,50)
p4=GPIO.PWM(OUT4,50)
p1.start(0)
p2.start(0)
p3.start(0)
p4.start(0)
def error_calc(cx_value, error_sum_val, prev_er_val):
    
    Kp=0.5  ##============================PID values ======================================
    Ki=0.005  ##0.05
    Kd=0.5  ##1
    error = 79 - cx_value   #middle x point 100
    prop_comp = error*Kp
    new_er_sum = error + error_sum_val
    int_comp = new_er_sum*Ki
    der_comp = (error-prev_er_val)*Kd
    PID_val = prop_comp + int_comp + der_comp
    return PID_val, new_er_sum, error
    
base_speed=20
er_sum=0
er_prev=0
try:
    time.sleep(2) 
    while True:
        
        # Display camera input
        image = picam2.capture_array("main")
        cv2.imshow('img',image)
    
        # Crop the image
        crop_img = image[60:240, 200:360]
        #[200:300, 225:425] gets it centered, sometimes has a hard time seeling the line, max cx=155ish9999999999999999
    
        # Convert to grayscale
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    
        # Gaussian blur
        blur = cv2.GaussianBlur(gray,(5,5),0)
        #cv2.imshow('blur',blur)
    
        # Color thresholding
        input_threshold,comp_threshold = cv2.threshold(blur,60,255,cv2.THRESH_BINARY_INV)
    
        # Find the contours of the frame
        contours,hierarchy = cv2.findContours(comp_threshold.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
        # Find the biggest contour (if detected)
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c) # determine moment - weighted average of intensities
            if int(M['m00']) != 0:
                cx = int(M['m10']/M['m00']) # find x component of centroid location
                cy = int(M['m01']/M['m00']) # find y component of centroid location
            else:
                print("Centroid calculation error, looping to acquire new values")
                continue

            cv2.line(crop_img,(cx,0),(cx,720),(255,0,0),1) # display vertical line at x value of centroid
            cv2.line(crop_img,(0,cy),(1280,cy),(255,0,0),1) # display horizontal line at y value of centroid
            cv2.drawContours(crop_img, contours, -1, (0,255,0), 2) # display green lines for all contours
            PID, er_sum, er_prev = error_calc(cx, er_sum, er_prev)
            print('PID:',PID)

            if PID > 40:
                continue
            
            if PID < -40:
                continue
              
            if cx >= 120:
                print("Turn Right")
                p1.ChangeDutyCycle(base_speed+(base_speed*(PID/40)))
                p3.ChangeDutyCycle(base_speed-(base_speed*(PID/40)))


            if cx < 120 and cx > 50:
                print("On Track!")
                p1.ChangeDutyCycle(base_speed)
                p3.ChangeDutyCycle(base_speed)

    
            if cx <= 50:
                print("Turn Left!")
                p1.ChangeDutyCycle(base_speed+(base_speed*(PID/40)))
                p3.ChangeDutyCycle(base_speed-(base_speed*(PID/40)))
              
            PID, er_sum, er_prev = error_calc(cx, er_sum, er_prev)
            print('PID:',PID)
            #time.sleep(0.1)
        else:
            print("I don't see the line")

    
        # Display the resulting frame
        cv2.imshow('frame',crop_img)
        

        # Show image for 1 ms then continue to next image
        cv2.waitKey(1)
except KeyboardInterrupt:
    print('All done')
    full_stop()

