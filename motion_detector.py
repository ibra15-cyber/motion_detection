import cv2
import time
from datetime import datetime
import pandas


# video=cv2.VideoCapture(0) #here int for webcame, more than 1 webcam means 0, 1, 2, etc.
video=cv2.VideoCapture("video.mp4") #for a video file #my webcam isnt working


#track number of frame
fr=0
first_frame=None
status_list=[None, None] #making a list to track status: either an obj enters or not
times=[] #a list to hold time of entering and leaving
df = pandas.DataFrame(columns=["Start", "End"]) #creating a dataframe obj with colum start and end

while True:
    fr+=1
    status=0 #we want to track this numbers
    check, frame = video.read()
    print(check) 
    print(frame)
    # time.sleep() 

    #to use a gray object
    gray_img=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    

    #adding gaussian blur the image
    gray_img=cv2.GaussianBlur(gray_img, (21,21), 0)

    if first_frame is None: #if first frame does not exit
        first_frame = gray_img #make gray_img as first frame and continue
        continue
    delta_frame =cv2.absdiff(first_frame, gray_img) #find the diff bet two images
    thresh_frame=cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1] #makes obj white
    thresh_frame=cv2.dilate(thresh_frame,None,iterations=2)
    
    (cnts,_)=cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue

        status=1 #if it succeeds in detecting a contor our variable is made 1
        
        (x,y,w,h)=cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3) #we expecting a rectangle above the body so we use the colored

    
    status_list.append(status) #append either 0 or 1, if an obj is detected 1 else 0
    
    if status_list[-1] == 1 and status_list[-2]==0:
        times.append(datetime.now()) # if the status is object to no obj record the time
    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now()) # if the status code changes from empty to detection
    # else:
    #     times.append(datetime.now())
    #show captured frame OR its gray_img alternative
    #pass the name in here
    frame1=cv2.imshow("Colored frame", frame)
    f_frame=cv2.imshow("First image", first_frame)
    d_frame=cv2.imshow("Comparing", delta_frame)
    th_frame=cv2.imshow("Threshold", thresh_frame) 
    

    
    key=cv2.waitKey(1) #show the next frame in  1000 millisecond
    #incase you want to stop the frame make it zero
    #else make it any number you want it to keep changing
    #1 millisecond is the fastest
    
    #key is expecting a keybord
    #if 0 was passed it quits
    #any other numb here it goes to the next frame
    #because of our while loop
    #so we want to the q unicode value and tell it to break
   
    if key==ord('q'): #ord is very useful
        if status==1:
            times.append(datetime.now() ) #means when i quit it should record the final time
        break

print(fr)
print(status_list) #we will see 0 when no obj enters but one when an obj does
print(times)
for time in range(0, len(times), 2): #iterating time list of both 0 and 1 or obj appearing and not to create a  df
    df=df.append(
        {
            "Start":times[time],  #the first time which is detection
            "End":times[time+1],   #the second which is off 
        },
        ignore_index=True #dictionary doesnt care about indexing
    ) 

df.to_csv("Times.csv") #saving our file to csv
video.release()
cv2.destroyAllWindows

