import math
import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpPose = mp.solutions.pose
pose = mpPose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
mpDraw = mp.solutions.drawing_utils
faceava = False
bodyava = False
text = ""
Reye = [0,0,0]
r1 = 0
r2 = 0
r3 = 0
d1 = False
d2 = False
d3 = False
right = False
left = False
mainangle = 0
facedriection = 0
light1 = False
light2 = False
light3 = False
light4 = False
light5 = False
light6 = False
point_nose = [0,0,0]
point_Leye = [0,0,0]
point_Reye = [0,0,0]
point_Lshoulder = [0,0,0]
point_Rshoulder = [0,0,0]
point_Lhip = [0,0,0]
vec_1=[0,0,0]
vec_3=[0,0,0]
x_vec = [0,0]
x2_vec = [0,0]
vec_2 = [0,0,0]
vec_4=[0,0,0]
norm = [0,0,0]
norm2= [0,0,0]
x_axis = [1,0]
x2_axis = [1,0]
L_M = []
angle = 0
angle2=0
success, image2 = cap.read()

while cap.isOpened():
    #success, image = cap.read()
    success, image1 = cap.read()
    #success, image2 = cap.read()

    cx = 0
    cx2 = 0
    image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)
    results1 = pose.process(image)

    img_h, img_w, img_c = image.shape
   
    differntOfFrames = cv2.absdiff(image1,image2)

    gray = cv2.cvtColor(differntOfFrames, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    text = "Motion detection"
  
    for contour in contours:
        

        if (cv2.contourArea(contour) > 500) :
           #turn on all the lights
            if (not faceava) and (not bodyava) :
                light1 = True
                light2 = True
                light3 = True
                light4 = True
                light5 = True
                light6 = True
                print ("All lights on")
            
            faceava = False
            bodyava = False

            if results.multi_face_landmarks:
                faceava = True
                for face_landmarks in results.multi_face_landmarks:
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx == 33 or idx == 263 or idx == 1:
                                cx,cy,cz = int(lm.x * img_w),int(lm.y*img_h),int(lm.z*-8000)
                                cv2.circle(image, (cx,cy),5,(255,34,34),cv2.FILLED)
                                if cx==0:
                                    cx = 1
                                if cy ==0:
                                    cy = 1
                                if idx == 1:
                                        point_nose[0]=(cx)
                                        point_nose[1]=(cy)
                                        point_nose[2]=(cz)
                                        #print (cx)
                                if idx == 33:
                                        point_Leye[0]=(cx)
                                        point_Leye[1]=(cy)
                                        point_Leye[2]=(cz)      
                                if idx == 263:
                                        point_Reye[0]=(cx) 
                                        point_Reye[1]=(cy) 
                                        point_Reye[2]=(cz)
                                        Reye = point_Reye  

                #print ("move with face Angle - ",angle)
                for i in range(3):
                    vec_1[i]=point_Reye[i]-point_Leye[i]
                mg = math.sqrt((vec_1[0]**2) + (vec_1[1]**2) + (vec_1[2]**2))
                # print (mg)
                #turn on correct light

            elif results1.pose_landmarks:
                bodyava = True
                for idx2, lm2 in enumerate(results1.pose_landmarks.landmark):
                    if idx2 == 11 or idx2 == 12 or idx2 == 23 or idx2 == 3:
                            cx2,cy2,cz2 = int(lm2.x * img_w/10),int(lm2.y*img_h/10),int(lm2.z*-200)
                            cv2.circle(image, (cx2*10,cy2*10),5,(255,34,34),cv2.FILLED) 
                            if cx2==0:
                                cx2 = 1
                            if cy2 ==0:
                                cy2 = 1
                            if idx2 == 11:
                                    point_Lshoulder[0]=(cx2)
                                    point_Lshoulder[1]=(cy2)
                                    point_Lshoulder[2]=(cz2)
                                    #cv2.circle(image, (cx,cy),10,(255,34,34),cv2.FILLED)
                                    #print ("Left   - ",cx,cy,cz)
                            if idx2 == 12:
                                    point_Rshoulder[0]=(cx2)
                                    point_Rshoulder[1]=(cy2)
                                    point_Rshoulder[2]=(cz2)     
                                    #cv2.circle(image, (cx,cy),10,(50,34,225),cv2.FILLED) 
                            if idx2 == 23:
                                    point_Lhip[0]=(cx2) 
                                    point_Lhip[1]=(cy2) 
                                    point_Lhip[2]=(cz2) 
                            if idx2 == 3:
                                    Reye[0]=(cx2 * 10) 
                                    Reye[1]=(cy2 * 10) 
                                    Reye[2]=(cz2)
            
  
                
            else:        
                print("move & cant find human figure")
                light1 = False
                light2 = False
                light3 = False
                light4 = False
                light5 = False
                light6 = False 
                #turn off all the lights  
            r0 = Reye[0]
            r1 = Reye[1]
            r2 = Reye[2]

            #cv2.circle(image, (r0,r1),9,(25,144,94),cv2.FILLED)        
    
    if faceava :    
        text = "Face detection"
        for i in range(3):
            vec_1[i]=point_nose[i]-point_Leye[i]
        
        for i in range(3):
            vec_2[i]=point_nose[i]-point_Reye[i]

        norm = np.cross(vec_1,vec_2)
        x_vec[0]=norm[0]
        x_vec[1]=norm[2]
        mag = math.sqrt((x_vec[0]*x_vec[0]) + (x_vec[1]*x_vec[1]))
                
        angle = (np.arccos((np.dot(x_vec,x_axis))/(mag)))*57.3 
        #print ("move with face Angle - ",(180-angle))    
        facedriection = (180-angle)
    if bodyava :
        text = "Body detection"
        for i in range(3):
            vec_3[i]=point_Lhip[i]-point_Lshoulder[i]
        
        for i in range(3):
            vec_4[i]=point_Lhip[i]-point_Rshoulder[i]  

        norm2 = np.cross(vec_3,vec_4)
        x2_vec[0]=norm2[0]
        x2_vec[1]=norm2[2] 

        mag2 = math.sqrt((x2_vec[0]*x2_vec[0]) + (x2_vec[1]*x2_vec[1]))
        #print (s,s*s,(x_vec[1]*x_vec[1]))
        #print ((x_vec[0]*x_vec[0]) + (x_vec[1]*x_vec[1]))
        angle2 = (np.arccos((np.dot(x2_vec,x2_axis))/(mag2)))*57.3                                                      
        #print("move with human figure - ",angle2)
        facedriection = (angle2)

    if light1 :
        cv2.circle(image, (50,50),10,(255,0,0),cv2.FILLED)
    if not light1 :
        cv2.circle(image, (50,50),10,(255,255,0),cv2.FILLED)
    if light2 :
        cv2.circle(image, (150,50),10,(255,0,0),cv2.FILLED)
    if not light2 :
        cv2.circle(image, (150,50),10,(255,255,0),cv2.FILLED)
    if light3 :
        cv2.circle(image, (50,150),10,(255,0,0),cv2.FILLED)
    if not light3 :
        cv2.circle(image, (50,150),10,(255,255,0),cv2.FILLED)
    if light4 :
        cv2.circle(image, (150,150),10,(255,0,0),cv2.FILLED)
    if not light4 :
        cv2.circle(image, (150,150),10,(255,255,0),cv2.FILLED)
    if light5 :
        cv2.circle(image, (50,250),10,(255,0,0),cv2.FILLED)
    if not light5 :
        cv2.circle(image, (50,250),10,(255,255,0),cv2.FILLED)
    if light6 :
        cv2.circle(image, (150,250),10,(255,0,0),cv2.FILLED)
    if not light6 :
        cv2.circle(image, (150,250),10,(255,255,0),cv2.FILLED)

    cv2.putText(image, "Mode- "+text, (180, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow('Head Pose Estimation', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image2=image1
    #success,image2 = cap.read()
    # print (facedriection,r1)
    
    if facedriection > 174 :
         right = False
         left = True
         #print ("Facing - Right")
    if facedriection < 6 :
         right = True
         left = False
         #print ("Facing - Left")
    if 6 < facedriection < 174 :
         right = False 
         left = False
         #print ("Facing - Front")
    if 10 < r1 < 170:
         d1 = True
         d2 = False
         d3 = False
         print ("Row 1")
    if 170 < r1 < 200:
         d1 = True
         d2 = True
         d3 = False
         print ("Row 1 & Row 2")
    if 200 < r1 < 230:
         d1 = False
         d2 = True
         d3 = False
         print ("Row 2")
    if 230 < r1 < 260:
         d1 = False
         d2 = True
         d3 = True
         print ("Row 2 & Row 3")
    if 260 < r1 < 350:
         d1 = False
         d2 = False
         d3 = True
         print ("Row 3")
    
    if (right and (faceava or bodyava)) :
        light2 = False
        light4 = False
        light6 = False
        if d1 and (not d2) :
             light1 = True
             light3 = False
             light5 = False
        if d1 and d2 :
             light1 = True
             light3 = True
             light5 = False   
        if (d2 and (not d1)) and (not d3) :                      
             light1 = False
             light3 = True
             light5 = False          
        if d2 and d3 :
             light1 = False
             light3 = True
             light5 = True
        if d3 and (not d2) :
             light1 = False
             light3 = False
             light5 = True
      
    if (left and (faceava or bodyava)) :
        light1 = False
        light3 = False
        light5 = False
        if d1 and (not d2) :
             light2 = True
             light4 = False
             light6 = False
        if d1 and d2 :
             light2 = True
             light4 = True
             light6 = False   
        if (d2 and (not d1)) and (not d3) :                      
             light2 = False
             light4 = True
             light6 = False          
        if d2 and d3 :
             light2 = False
             light4 = True
             light6 = True
        if d3 and (not d2) :
             light2 = False
             light4 = False
             light6 = True
    
    if ((not right) and (not left)) and (faceava or bodyava) :
        #print ("front")
        if d1 and (not d2) :
             light1 = True
             light2 = True
             light3 = False
             light4 = False
             light5 = False
             light6 = False
        if d1 and d2 :
             light1 = True
             light2 = True
             light3 = True
             light4 = True
             light5 = False
             light6 = False
        if (d2 and (not d1)) and (not d3) :
             light1 = False
             light2 = False
             light3 = True
             light4 = True
             light5 = False
             light6 = False
        if d2 and d3 :
             light1 = False
             light2 = False
             light3 = True
             light4 = True
             light5 = True
             light6 = True
        if d3 and (not d2) :
             light1 = False
             light2 = False
             light3 = False
             light4 = False
             light5 = True
             light6 = True
   

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()