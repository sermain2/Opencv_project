import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height
faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

face_id = input("\n 사용자의 번호를 입력해주세요. <return> ==> ")
print("\n [INFO] 얼굴 사진을 초기화 중입니다. 카메라를 보고 기다려주세요.")

# 개별 샘플링 얼굴 수 초기화
count = 200
while True:
    ret, img = cam.read()  # img = cv2.flip(img, 1) # 상하반전
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 이후 얼굴을 검출할 gray scale을 만듦
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        # 이미지 피라미드에 사용하는 scalefactor
        # scale 안에 들어가는 이미지의 크기가 1.2씩 증가 즉 scale size는 그대로
        # 이므로 이미지가 1/1.2 씩 줄어서 scale에 맞춰지는 것이다.
        minNeighbors=5,
        # 최소 가질 수 있는 이웃으로 3~6사이의 값을 넣어야 detect가 더 잘된다고 한다.
        # Neighbor이 너무 크면 알맞게 detect한 rectangular도 지워버릴 수 있으며,
        # 너무 작으면 얼굴이 아닌 여러개의 rectangular가 생길 수 있다.
        # 만약 이 값이 0이면, scale이 움직일 때마다 얼굴을 검출해 내는 rectangular가 한 얼굴에
        # 중복적으로 발생할 수 있게 된다.
        minSize=(20, 20)
        # 검출하려는 이미지의 최소 사이즈로 이 크기보다 작은 object는 무시
        # maxSize도 당연히 있음.
    )
    for x, y, w, h in faces:
        # 좌표 값과 rectangular의 width height를 받게 된다.
        # x,y값은 rectangular가 시작하는 지점의 좌표
        # 원본 이미지에 얼굴의 위치를 표시하는 작업을 함.
        # for문을 돌리는 이유는 여러 개가 검출 될 수 있기 때문.
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1
        # 사진을 찍고 데이터 셋 폴더에 저장
        cv2.imwrite(
            "dataset/User." + str(face_id) + "." + str(count) + ".jpg",
            gray[y : y + h, x : x + w],
        )
        cv2.imshow("image", img)
    k = cv2.waitKey(100) & 0xFF
    if k == 27:
        break
    elif count >= 300:
        break
print("\n [INFO] 프로그램 종료.")
cam.release()
cv2.destroyAllWindows()
