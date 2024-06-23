import numpy as np
import os, cv2
from PIL import Image
def train_classifier(data_dir):
    path=[os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    faces=[]
    ids=[]
    
    for image in path:
        img=Image.open(image).convert('L') # opening an image and converting it into greyscale image
        imageNp= np.array(img, 'uint8') #img into numpy for classifier
        id= int(os.path.split(image)[1].split(".")[1])

        faces.append(imageNp)
        ids.append(id)
    
    ids= np.array(ids)

    clf= cv2.face.LBPHFaceRecognizer_create() #feed to classifier and train it
    clf.train(faces, ids)
    clf.write("classifier.yml")

train_classifier("data")