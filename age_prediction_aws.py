import numpy as np
import cv2
import boto3
import json
import csv
import time
font = cv2.FONT_HERSHEY_SIMPLEX
fontColor = (0, 255, 255)
fontSize = 0.8
haar_file = 'haarcascade_frontalface_default.xml'


# Part 2: Use fisherRecognizer on camera stream
face_cascade = cv2.CascadeClassifier(haar_file)


with open("new_user_credentials.csv",'r') as input:
    next(input)
    reader=csv.reader(input)
    print(reader)

    for line in reader:
        access_key_id=line[2]
        secret_key_id=line[3]

# Rekognition Detect faces
def detect_faces(photo):

    # client=boto3.client('rekognition')
    region = "eu-west-1"
    client = boto3.client("rekognition", aws_access_key_id=access_key_id,
                          aws_secret_access_key=secret_key_id, region_name=region)

    response = client.detect_faces(
        Image={
            'Bytes': photo
        },
        Attributes=[
            'ALL'
        ]
    )
    return response

cam = cv2.VideoCapture(0)
cv2.namedWindow("Face Predictor")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

while True:
    ret, frame = cam.read()

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Face Predictor", frame)
    print(frame.shape)
    out.write(frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        resp=(detect_faces(cv2.imencode('.jpg', frame)[1].tostring()))
        print(resp)

        AgeL = resp['FaceDetails'][0]['AgeRange']['Low']
        AgeH = resp['FaceDetails'][0]['AgeRange']['High']
        print((AgeL, AgeH))

        print('Age : ' + '{0} - {1}'.format(AgeL, AgeH))
        Gender = resp['FaceDetails'][0]['Gender']['Value']
        print(str(Gender))

        Beard = resp['FaceDetails'][0]['Beard']['Value']
        print(str(Beard))



        Eyeglasses = resp['FaceDetails'][0]['Eyeglasses']['Value']
        print(str(Eyeglasses))

        cv2.putText(frame, "Gender : " + Gender, (10, 30), font, fontSize, fontColor, 2)
        cv2.putText(frame, 'Age : ' + '{0} - {1}'.format(AgeL, AgeH), (10, 60), font, fontSize, fontColor, 2)
        if Beard:
            cv2.putText(frame, "  Beard " , (10, 90), font, fontSize, fontColor, 2)
        if Eyeglasses:
            cv2.putText(frame, "Eyeglasses " , (10, 120), font, fontSize, fontColor, 2)



        cv2.imshow("Face Predictor", frame)
        #
        for i in range(0, 40):
            out.write(frame)
        cv2.waitKey(2000)
        # time.sleep(2)





out.release()

# Release control of the webcam and close window
cam.release()
cv2.destroyAllWindows()