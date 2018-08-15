#!/usr/bin/env python
# -*- coding: utf-8 -*
import cv2
import serial
import time
ser = serial.Serial("/dev/ttyAMA0", 9600)  # 打开串口

def uart():
    while True:
        count = ser.inWaiting()  # 获得接收缓冲区字符
        if count != 0:
            recv = ser.read(count)  # 读取内容,一次读10个字节
            bytes.decode(recv)  # 字节转成字符
            print(recv)  # 打印接收的数据
            return recv
        else:
            break
        ser.flushInput()  # 清空接收缓冲区
        time.sleep(0.1)  # 必要的软件延时

cap = cv2.VideoCapture(0)
acount = 0
bcount = 0
ccount = 0
dcount = 0
ecount = 0
fcount = 0
while(True):
    ret, frame = cap.read()
    cv2.namedWindow("capture", 0)
    cv2.resizeWindow("capture", 640, 480)
    cv2.imshow("capture", frame)
    cv2.waitKey(1)
    key = uart()
    if key == 'a':
        cv2.imwrite("/home/pi/Downloads/data/battery/battery%d.jpeg"%acount, frame)
        acount += 1
        ser.write("aok ")
    elif key=='b':
        cv2.imwrite("/home/pi/Downloads/data/bottle/bottle%d.jpeg"%bcount, frame)
        bcount += 1
        ser.write("bok ")
    elif key == 'c':
        cv2.imwrite("/home/pi/Downloads/data/bowl/bowl%d.jpeg"%ccount, frame)
        ccount += 1
        ser.write("cok ")
    elif key == 'd':
        cv2.imwrite("/home/pi/Downloads/data/box/box%d.jpeg"%dcount, frame)
        dcount += 1
        ser.write("dok ")
    elif key == 'e':
        cv2.imwrite("/home/pi/Downloads/data/milk-box/milk-box%d.jpeg"%ecount, frame)
        ecount += 1
        ser.write("eok ")
    elif key == 'f':
        cv2.imwrite("/home/pi/Downloads/data/paper/paper%d.jpeg"%fcount, frame)
        fcount += 1
        ser.write("fok ")
    elif key == 'q':
        break
cap.release()
cv2.destroyAllWindows()


