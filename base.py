import cv2
import numpy as np
import face_recognition
#Step1: import images and convert them into RGB
#Step2: encoding
#Step3: comparing images

klausPic = face_recognition.load_image_file("Basic/klaus1.jpg")
klausPic = cv2.cvtColor(klausPic,cv2.COLOR_BGR2RGB)

klausTest = face_recognition.load_image_file("Basic/park.jpg")
klausTest = cv2.cvtColor(klausTest,cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(klausPic)[0]
encodeKlaus = face_recognition.face_encodings(klausPic)[0]
cv2.rectangle(klausPic,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

faceloc2 = face_recognition.face_locations(klausTest)[0]
encodeKlaus2 = face_recognition.face_encodings(klausTest)[0]
cv2.rectangle(klausTest,(faceloc2[3],faceloc2[0]),(faceloc2[1],faceloc2[2]),(255,0,255),2)

compare = face_recognition.compare_faces([encodeKlaus],encodeKlaus2)
face_dist = face_recognition.face_distance([encodeKlaus],encodeKlaus2)
print(compare,face_dist)
cv2.putText(klausTest,f'{compare} {round(face_dist[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow("KLAUS",klausPic)
cv2.imshow("KLAUS TEST",klausTest)
cv2.waitKey(0)


