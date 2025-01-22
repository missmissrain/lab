import cv2
import torch
from ultralytics import YOLO

model = YOLO("I:\\best.pt").to('cuda')#模型文件
stream_url = "rtsp://@Yiiiii:ctn041011@192.168.31.30:554" #视频流
targe_classes = [0] #目标类别

cap = cv2.VideoCapture(stream_url) # 打开视频流

if not cap.isOpened():
    print("无法打开视频流")
    exit()

print("视频流已打开")

while True:
    ret, frame = cap.read()# 读取视频帧
    if not ret:
        print("无法读取视频帧")
        break
    results = model.predict(source=frame,classes = targe_classes)# 预测


    now_frame = results[0].plot()# 绘制结果

    cv2.imshow("frame", now_frame)# 显示结果
    if cv2.waitKey(1) & 0xFF == ord('q'):#输入q结束
        break

cap.release()
cv2.destroyAllWindows()
