#!/usr/bin/env python
# -*- coding: utf-8 -*
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import glob,serial,time,os
import cv2
from os import system

BOTTLENECK_TENSOR_SIZE = 2048
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
import_BOTTLENECK_TENSOR_NAME = 'import/pool_3/_reshape:0'
import_JPEG_DATA_TENSOR_NAME = 'import/DecodeJpeg/contents:0'
CACHE_DIR = "/home/pi/Downloads/ince/lable/"
INPUT_DATA = "/home/pi/Downloads/ince/data/"       # 数据库存放地址

def create_image_lists(testing_percentage, validation_percentage):
    result = {}                                             # 保存所有图像。key为类别名称。value也是字典，存储了所有的图片名称
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]          # 获取所有子目录
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:                                     # 第一个目录为当前目录，需要忽略
            is_root_dir = False
            continue                                        # 繼續讀取子目錄

        # 获取当前目录下的所有有效图片
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']

        file_list = []                                      # 存储所有图像
        dir_name = os.path.basename(sub_dir)                # 获取路径的最后一个目录名字
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list: continue

        # 将当前类别的图片随机分为训练数据集、测试数据集、验证数据集
        label_name = dir_name.lower()                       # 通过目录名获取类别的名称

        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)        # 获取该图片的名称

            # 随机划分数据
            chance = np.random.randint(100)                # 随机产生100个数代表百分比
            if chance < validation_percentage:
                validation_images.append(base_name)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)

        # 将当前类别的数据集放入结果字典
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }
    return result                                         # 返回整理好的所有数据

def uart():
    # ser.write("again\n".encode('utf-8'))  # 发送数据
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


def get_bottleneck_path_xiaojie(CACHE_DIR, image_path):
    file_name_suffix = image_path.split('/')[-1]
    file_name_no_suffix = file_name_suffix.split('.')[0]
    bottleneck_file_name = file_name_no_suffix + ('_cache.txt')
    bottleneck_path = os.path.join(CACHE_DIR, bottleneck_file_name)
    return bottleneck_path


def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


def get_bottleneck_values_xiaojie(sess, image_path, jpeg_data_tensor, bottleneck_tensor):
    # 求缓存文件的位置
    if not os.path.exists(CACHE_DIR): os.makedirs(CACHE_DIR)
    bottleneck_path = get_bottleneck_path_xiaojie(CACHE_DIR, image_path)
    # print (bottleneck_path)
    if not os.path.exists(bottleneck_path):
        image_data = gfile.FastGFile(image_path, 'rb').read()
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
            bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values


# 分类图片
def classify_photo(sess, jpeg_data_tensor, bottleneck_tensor):
    image_path = "/home/pi/Downloads/ince/picture/opencv.jpeg"
    image_data = gfile.FastGFile(image_path, 'rb').read()
    print (sess.run(jpeg_data_tensor,{jpeg_data_tensor:image_data}))
    print (sess.run(bottleneck_tensor,{jpeg_data_tensor:image_data}))
    # """
    bottleneck_input = sess.graph.get_tensor_by_name("BottleneckInputPlaceholder:0")
    final_tensor = sess.graph.get_tensor_by_name("final_training_ops/Softmax:0")
    bottleneck = sess.run(bottleneck_tensor, feed_dict={jpeg_data_tensor: image_data})
    class_result = sess.run(final_tensor, feed_dict={bottleneck_input: bottleneck})
    images_lists = create_image_lists(5, 5)
    a = np.squeeze(class_result)
    print(a)
    r = np.max(class_result)
    print(r)

    classes = ['battery', 'box', 'milk-box', 'bottle','bowl', 'paper']

    if r >= 0.60:
        kinds = int(np.argwhere(a==np.max(a)))
        print(kinds)
        print(classes[int(np.argwhere(a==np.max(a)))])
        if kinds == 0:
            ser.write("a12")
        elif kinds == 1:
            ser.write("b12")
        elif kinds == 2:
            ser.write("c12")
        elif kinds == 3:
            ser.write("d12")
        elif kinds == 4:
            ser.write("e12")
        elif kinds == 5:
            ser.write("f12")
    else:
        print("此物品不在分类物品当中")
        ser.write("g12")

# 加载模型
saver = tf.train.import_meta_graph("/home/pi/Downloads/ince/module/modle/model1.ckpt.meta")
with tf.Session() as sess:
    saver.restore(sess, "/home/pi/Downloads/ince/module/modle/model1.ckpt")
    bottleneck_tensor = sess.graph.get_tensor_by_name(import_BOTTLENECK_TENSOR_NAME)
    jpeg_data_tensor = sess.graph.get_tensor_by_name(import_JPEG_DATA_TENSOR_NAME)
    # all*
    ser = serial.Serial("/dev/ttyAMA0", 9600)  # 打开串口
    cap = cv2.VideoCapture(0)  #camera init
    while (1):
        key = uart()
        ret, frame = cap.read()  # 读取一帧
        cv2.namedWindow("capture", 0)
        cv2.resizeWindow("capture",640,480)
        cv2.imshow("capture", frame)
        cv2.waitKey(1)
        if key == 'a':
            ret, frame = cap.read()  # 读取一帧
            cv2.imwrite("/home/pi/Downloads/ince/picture/opencv.jpeg", frame)  # 写入图片
            classify_photo(sess, jpeg_data_tensor, bottleneck_tensor)
        elif key == 'q':
            ser.write("shutdown-ing")
            ser.close()  # 释放串口
            sess.close()
            cap.release()  # 释放摄像头
            #system("shutdown -t 0")
            break
        elif key== 'r':
            ser.write("reboot-ing")
            ser.close()  # 释放串口
            sess.close()
            cap.release()  # 释放摄像头
            system('reboot')
            break