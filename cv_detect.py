import os 
import time
import cv2 as cv
from cv2 import *
import numpy as np

#define
ERROR = -1

#define threadhold
Conf_threshold = 0.3
NMS_threshold = 0.4

#function will read all class name in class_file and return a list[]
def read_classes(class_file):
    #Read class name from class.txt and save to array class
    class_name = []
    with open(class_file, 'r') as f:
        class_name = [cname.strip() for cname in f.readlines()]
    print(class_name)
    return class_name

#function use to load model yolov4 to detect
#return model
def load_model_yolov4(weights,cfg):
    #load model yolov4
    net = cv.dnn.readNet(weights,cfg)
    #set handle with cuda
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
    model = cv.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
    print("loading model: " + weights +" with " + cfg)
    return model

#capture image input from camera or load image from storge
#note: /dev/video0 => videox = 0
def get_image_from_cam(videox):
    cam = VideoCapture(videox)
    result, image = cam.read()
    if result:
        print("capture image from camera success")
        return image
    else:
        print("capture image from camera failed !")
        return ERROR

#load image test from stogre
def load_image_local(link):
    image = cv.imread(link)
    return image


#main function begin here
def main():
    print("running...")
    my_model = load_model_yolov4('yolov4-tiny-custom_best.weights','yolov4-tiny-custom.cfg')
    img = get_image_from_cam(0)
    img = load_image_local('./test/OOS/img2.jpg')
    classes, score, box = my_model.detect(img, Conf_threshold, NMS_threshold)
    print(classes)
    print(box)
    print(score)
    #cv.imshow(img)
    #cv.imwrite('test.jpg',img)

    #loop
    #while True:
        



if __name__ == "__main__":
    main()