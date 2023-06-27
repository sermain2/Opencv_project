import cv2
import numpy as np
from PIL import Image
import os

path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create() #LBP알고리즘을 이용하기 위한 새 변수를 생성
detector = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml");

# 이미지를 불러와서 라벨링 하기
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)] # 이미지 파일 안에 
    faceSamples=[] #각 이미지의 얼굴 값을 array uint8 형태로 저장한것을 dictionary 형태로 저장
    ids = [] #여러 개의 아이디 값을 배열로 저장
    for imagePath in imagePaths: #이미지 파일을 하나씩 받아 옴
        PIL_img = Image.open(imagePath).convert('L') # image를 grayscale로 변환 시킴
        img_numpy = np.array(PIL_img,'uint8') #np.array로 img 파일을 int형으로 변환시켜 저장
        id = int(os.path.split(imagePath)[-1].split(".")[1]) # 파일의 아이디 추출
        faces = detector.detectMultiScale(img_numpy) # 다시 얼굴 이미지에서 또 얼굴을 추출(얼굴의 크기를 알기 위함)
        
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w]) #img를 int형 으로 바꾼 sample들을 넣은 배열
            ids.append(id) #id값을 쭉 넣어서 배열로 만듦
    return faceSamples,ids

print ("\n [INFO] 얼굴을 학습 중 입니다. 잠시만 기다려 주세요.")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids)) #LBP matrix를 만듦

# trainer.yml로 모델 저장히기
recognizer.write('trainer/trainer.yml') #만든 LBP matrix를 yml 파일 형태로 저장

# 학습 된 얼굴 수 뽑고, 종료 하기
print("\n [INFO] {0} 학습 완료, 프로그램을 종료합니다.".format(len(np.unique(ids)))) #ids 배열의 개수만큼 훈련되었다고 표시함.

