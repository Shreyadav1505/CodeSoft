import cv2
import os
#chmod +r classifier.yml

def generate_dataset(img, id, img_id):
    cv2.imwrite("data/user."+str(id)+"."+str(img_id)+".jpeg",img)

def draw_boundary(img, classifier, scaleFactor, minNeighbours, color, text): #personalised classifier for your generated dataset
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features= classifier.detectMultiScale(gray_img, scaleFactor, minNeighbours)
    coords=[]

    for (x, y, w, h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        if id==1:
            cv2.putText(img, "Shreya", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords =[x, y, w, h]
    return coords


def detect(img, faceCascade, eyesCascade, noseCascade, mouthCascade, img_id):
    color= {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0), "black":(0,0,0)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, color['blue'], "Face")

    if len(coords)==4:
        roi_img= img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]  # used to give cordinates only for the face space. other entire surrounding in cut off.
        user_id=1
        generate_dataset(roi_img, user_id, img_id)
        # coords = draw_boundary(roi_img, eyesCascade, 1.1, 14, color['red'], "Eyes")  #roi= region of interest
        # coords = draw_boundary(roi_img, mouthCascade, 1.1, 20, color['green'], "Mouth")
        # coords = draw_boundary(roi_img, noseCascade, 1.1, 20, color['black'], "Nose")   # can be used to detect mouth, eyes and nose on the face.
    return img



faceCascade= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyesCascade=cv2.CascadeClassifier("haarcascade_eye.xml")
mouthCascade=cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")
noseCascade=cv2.CascadeClassifier("haarcascade_mcs_nose.xml")



video_capture=cv2.VideoCapture(0)  #0 for laptop camera, -1 for external
img_id=0

while True:
    _, img = video_capture.read()
    img = detect(img, faceCascade, eyesCascade, noseCascade, mouthCascade, img_id)
    cv2.imshow("face detection", img)
    img_id+=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
