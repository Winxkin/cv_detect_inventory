import os 
import time
import cv2 as cv
from cv2 import *
import numpy as np
import firebase_admin as firebase
from twilio.rest import Client


#define
ERROR = -1

#define threadhold
Conf_threshold = 0.3
NMS_threshold = 0.4
COLORS = [(0, 255, 0)]

#SMS
Huan_phone = '+84866078421'


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
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
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

#get target from model detect and return amount of object
def get_detection(class_name,classids, scores, boxes, *img):
    amount = 0
    for (classid, score, box) in zip(classids, scores, boxes):
        cv.rectangle(*img,(box[0],box[1]), (box[0] + box[2],box[1] + box[3]),
        color=(0, 0, 255),thickness=2)
        label = "%s : %f" % (class_name[classid],score)
        cv.putText(*img, label,(box[0], box[1]-5), cv.FONT_HERSHEY_SIMPLEX, 1,
        color=(0, 0, 255) ,thickness=2)
        amount = amount + 1

    print("total empty position: " + str(amount))
    return amount

#send SMS to phone
def sendSMS(phone, msg):
    # Your Account Sid and Auth Token from twilio.com / console
    account_sid = 'ACf4c77bd7c8472817692e7b5313c0cb8b'
    auth_token = '8aed6b0fb5cf7f6835fb0adc0c92ed85'
    _client = Client(account_sid, auth_token)
    message = _client.messages.create(
                              from_='+15673716123',
                              body = msg,
                              to = phone
                          )
  
    print(message.sid)
    return


#main function begin here
def main():
    print("running...")
    class_name = read_classes('class.txt')
    my_model = load_model_yolov4('yolov4-tiny-custom_best.weights','yolov4-tiny-custom.cfg')
    #img = get_image_from_cam(0)
    img = load_image_local('./test/OOS/img1.jpg')
    classids, scores, boxes = my_model.detect(img, Conf_threshold, NMS_threshold)
    print(classids)
    print(boxes)
    print(scores)
    count = get_detection(class_name,classids,scores,boxes,img)

    #print("total empty position" + amount)
    
    cv.imwrite('test.jpg',img)

    #loop
    #while True:
        
def test():
    print("test ^^")
    #sendSMS(Huan_phone,'Shelf 101 had out of stock with 7 slots empty')
    


if __name__ == "__main__":
    test()
    #main()