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
    count = 0
    for (classid, score, box) in zip(classids, scores, boxes):
        #check if class is oos
        if classid == 0:
            cv.rectangle(*img,(box[0],box[1]), (box[0] + box[2],box[1] + box[3]),
            color=(0, 0, 255),thickness=2)
            #label = "%s" % (class_name[classid])
            #cv.putText(*img, label,(box[0], box[1]-5), cv.FONT_HERSHEY_PLAIN, 1,
            #color=(0, 0, 255) ,thickness=2)
        
        #check if class is in-stock
        if classid == 1:
            cv.rectangle(*img,(box[0],box[1]), (box[0] + box[2],box[1] + box[3]),
            color=(0, 255, 0),thickness=2)
            #label = "%s" % (class_name[classid])
            #cv.putText(*img, label,(box[0], box[1]-5), cv.FONT_HERSHEY_PLAIN, 1,
            #color=(0, 255, 0) ,thickness=2)

    return

#change id in list id of class object from 0 -> 1
def change_id_object(classids_obj):
    for index,id in enumerate(classids_obj):
        if classids_obj[index] == 0:
            classids_obj[index] = 1
    return

#merge bouding box data from OOS and Obj
def append_boxes(boxes_1,boxes_2):
    boxes = []
    for box in boxes_1:
        boxes.append(box)

    for box in boxes_2:
        boxes.append(box)
    return boxes

#send SMS to phone
def sendSMS(phone, msg):
    # Your Account Sid and Auth Token from twilio.com / console
    account_sid = 'ACf4c77bd7c8472817692e7b5313c0cb8b'
    auth_token = '9bb84200df110c0b90b557769bcf0120'
    _client = Client(account_sid, auth_token)
    message = _client.messages.create(
                              from_='+15673716123',
                              body = msg,
                              to = phone
                          )
  
    print(message.sid)
    return

#delay for x secconds
def delay_secconds(secconds):
    for s in range(secconds):
        time.sleep(1) #sleep in 1 seccond
        print("waitting....." + str(s) +"s")
    return

#main function begin here
def main():
    print("running...")
    class_name = read_classes('class.txt')
    OOS_model = load_model_yolov4('yolov4-tiny-OOS.weights','yolov4-tiny-custom-OOS.cfg')
    obj_model = load_model_yolov4('yolov4-tiny-obj.weights','yolov4-tiny-custom-obj.cfg')

    #while loop
    while True:
        #img = get_image_from_cam(0)
        img = load_image_local('./test/OOS/img15.jpg')
        classids_oos, scores_oos, boxes_oos = OOS_model.detect(img, Conf_threshold, NMS_threshold)
        classids_obj, scores_obj, boxes_obj = obj_model.detect(img, Conf_threshold, NMS_threshold)
        change_id_object(classids_obj)

        #get total of in-stock and OOS
        empty_total = len(classids_oos)
        obj_total = len(classids_obj)
        #state_shelf = (obj_total)/(empty_total + obj_total)

        #append detection beweent object and OOS
        classids = np.append(classids_oos,classids_obj)
        scores = np.append(scores_oos,scores_obj)
        boxes = append_boxes(boxes_oos,boxes_obj)

        #print classes id are detected
        print('classes id:')
        print(classids)

        #draw bouding box in image
        get_detection(class_name,classids,scores,boxes,img)

        #log total position
        print("total empty position : " + str(empty_total))
        print("total obj position : " + str(obj_total))
        #print("state of shelf: " + str(state_shelf) + '%')

        #sendSMS(Huan_phone,'Shelf 101 had out of stock with ' + str(empty_total) +' slots empty')
        cv.imwrite('test.jpg',img)
        delay_secconds(10)
    #end loop
    
        
def test():
    delay_secconds(10)
    return

if __name__ == "__main__":
    #test()
    main()