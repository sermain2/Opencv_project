#opencv를 이용하기 위한 모듈
import cv2
import numpy as np
import os

# 토글 스위치로 제어하기 위한 모듈
import RPi.GPIO as GPIO

# 데이터 베이스로 측정값을 보내기 위한 모듈
import MySQLdb
import pymysql

# 체온 즉정을 위한 모듈
import adafruit_mlx90614
import busio as io
import board
import time

# 21번 핀을 이용하여 카메라를 끌 수 있도록 토글 스위치를 장착
GPIO.setmode(GPIO.BCM)
GPIO.setup(21, GPIO.IN, pull_up_down = GPIO.PUD_DOWN)
sw_pin = 21

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('/home/pi/trainer/trainer.yml')
cascadePath = "/home/pi/haarcascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX #opencv에서 지원하는 font

#사용자 카운트
id = 0
# 이런식으로 사용자의 이름을 사용자 수만큼 추가해준다.
names = ['None', 'Semin', 'Jaehoon', 'Jaeseok']

# 이용할 데이터 베이스로 경로 설정
conn = pymysql.connect(host="220.69.207.85", user="min",passwd="1234",db="smartmirror")
i2c = io.I2C(board.SCL, board.SDA, frequency=100000)
mlx = adafruit_mlx90614.MLX90614(i2c)

# 실시간 비디오 캡처 초기화 및 시작
cam = cv2.VideoCapture(0)
cam.set(3, 640) # 비디오의 폭 설정 
cam.set(4, 480) # 비디오의 높이 설정

# 창 크기를 최소 크기로 정의하여 얼굴로 인식
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

try:
    with conn.cursor() as cur:
        sql="insert into smartmirror_sensor values(%s, %s, %s, %s);"

        while True:
            ret, img =cam.read()
            img = cv2.flip(img, 1) 
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            
            faces = faceCascade.detectMultiScale( 
                gray,
                scaleFactor = 1.2,
                minNeighbors = 6,
                minSize = (int(minW), int(minH)),
               )
            
            ambientString = "{:.1f}".format(mlx.ambient_temperature)
            objectString = "{:.1f}".format(mlx.object_temperature+10)
            
            cur.execute(sql,(None, ambientString, objectString,time.strftime("%Y-%m-%d %H:%M",time.localtime())))
            conn.commit() 
            
            for(x,y,w,h) in faces:
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

                if (70 <= confidence <= 100):
                    id = names[id]
                    confidence = "{0}%".format(round(confidence))
                    print(id + "체온:",objectString, "°C")
                    print("주변 온도:", ambientString, "°C")
                    time.sleep(0.4)
                    
                else:     
                    id = "unknown"
                    confidence = "{0}%".format(round(100-confidence))
                    
                #일치 확률과 이름을 화면에 출력
                cv2.putText(img, str(id), (x+5,y-5), font, 1, (0,255,0), 2)
                cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 2)  
                cv2.putText(img, str(objectString), (x+90,y+90), font, 1, (0,255,255), 2)
            cv2.imshow('camera',img)
            #최대한 자주 Key를 획득할 수 있도록 wait time을 줄임
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
            elif GPIO.input(sw_pin) == GPIO.HIGH:
                break
           
        # 작업 정리 
        print("\n [INFO] 프로그램 및 작업 정리 종료.")
        cam.release()
        cv2.destroyAllWindows()

except KeyboardInterrupt :
    exit()
finally:
    conn.close()
