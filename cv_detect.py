import os 
import time
import argparse
import cv2 as cv
from cv2 import *
import numpy as np
import firebase_admin
from firebase_admin import credentials
from firebase_admin import _http_client
from firebase_admin import storage, firestore
from firebase_admin import db
from twilio.rest import Client
from zalo.sdk.app import ZaloAppInfo, Zalo3rdAppClient

#define
ERROR = -1

#define threadhold
Conf_threshold = 0.3
NMS_threshold = 0.4
COLORS = [(0, 255, 0)]

#***********************************************************************
#function will read all class name in class_file and return a list[]
#***********************************************************************
def read_classes(class_file):
    #Read class name from class.txt and save to array class
    class_name = []
    with open(class_file, 'r') as f:
        class_name = [cname.strip() for cname in f.readlines()]
    print(class_name)
    return class_name

#***********************************************************************
#function use to load model yolov4 to detect
#return model
#***********************************************************************
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

#***********************************************************************
#capture image input from camera or load image from storge
#note: /dev/video0 => videox = 0
#***********************************************************************
def get_image_from_cam(videox):
    cam = cv.VideoCapture(videox)
    result, image = cam.read()
    if result:
        print("capture image from camera success")
        return image
    else:
        print("capture image from camera failed !")
        return ERROR

#***********************************************************************
#load image test from stogre
#***********************************************************************
def load_image_local(link):
    image = cv.imread(link)
    return image

#***********************************************************************
#get target from model detect and return amount of object
#***********************************************************************
def get_detection(class_name,classids, scores, boxes, *img):
    for (classid, score, box) in zip(classids, scores, boxes):
        #check if class is oos
        if classid == 0:
            cv.rectangle(*img,(box[0],box[1]), (box[0] + box[2],box[1] + box[3]),
            color=(0, 0, 255),thickness=2)
            label = "%s" % (class_name[classid])
            #cv.putText(*img, label,(box[0], box[1]-5), cv.FONT_HERSHEY_PLAIN, 1,
            #color=(0, 0, 255) ,thickness=1)
        
        #check if class is in-stock
        if classid == 1:
            cv.rectangle(*img,(box[0],box[1]), (box[0] + box[2],box[1] + box[3]),
            color=(0, 255, 0),thickness=1)
            label = "%s" % (class_name[classid])
            #cv.putText(*img, label,(box[0], box[1]-5), cv.FONT_HERSHEY_PLAIN, 1,
            #color=(0, 255, 0) ,thickness=1)

    return

#***********************************************************************
#get sum area bouding box of class in a picture
#***********************************************************************
def get_area_boxes(boxes):
    sum = 0
    for(box) in zip(boxes):
        weigh = box[0][2]
        high = box[0][3]
        sum = sum + (weigh*high)
    return sum

#***********************************************************************
#change id in list id of class object from 0 -> 1
#***********************************************************************
def change_id_object(classids_obj):
    for index,id in enumerate(classids_obj):
        if classids_obj[index] == 0:
            classids_obj[index] = 1
    return

#***********************************************************************
#merge bouding box data from OOS and Obj
#***********************************************************************
def append_boxes(boxes_1,boxes_2):
    boxes = []
    for box in boxes_1:
        boxes.append(box)

    for box in boxes_2:
        boxes.append(box)
    return boxes

#***********************************************************************
#send zalo sms to phone
#***********************************************************************
def zalo_SMS(user_id, msg):
    zalo_info = ZaloAppInfo(app_id="your app id", secret_key="your app secret")
    zalo_3rd_app_client = Zalo3rdAppClient(zalo_info)
    send_message = zalo_3rd_app_client.post('/me/message', token, {
    'message': msg,
    'to': user_id,
    'link': 'https://developers.zalo.me/'
    })
    return

#***********************************************************************
#delay for x secconds
#***********************************************************************
def delay_secconds(secconds):
    for s in range(secconds):
        time.sleep(1) #sleep in 1 seccond
        print("waitting....." + str(s) +"s")
    return

#***********************************************************************
#connect to firebase
#***********************************************************************
def connect_to_firebase():
    cred = credentials.Certificate("./inventory-firebase.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL' : 'https://inventory-16459-default-rtdb.asia-southeast1.firebasedatabase.app/',
        'storageBucket' : 'inventory-16459.appspot.com'
    })
    return

#***********************************************************************
#connecting to firebase realtime
#***********************************************************************
def post_to_firebaserealtime(OOS,In_stock,avalible,img_name):
    ref = db.reference('/')
    ref.set({
        'Shelf' : {
            'OOS' : OOS,
            'In-stock' : In_stock,
            'In-stock_avalivle' : avalible,
            'img_output' : img_name
        }
    })
    return

#***********************************************************************
#update image to firebase storage
#***********************************************************************
def upload_to_firebaseStorage(img_name):  
    db = firestore.client()
    bucket = storage.bucket()
    blob = bucket.blob(img_name)
    outfile = './' + img_name
    with open(outfile,'rb') as my_file:
        blob.upload_from_file(my_file)
    return
    
#*************************
#main function begin here
#*************************
def main():
    #add argument
    parser = argparse.ArgumentParser(description='project cv detection inventory.')
    parser.add_argument('--img',help='detect with input is image, directory at cv_detect_inventory/test/OOS')
    parser.add_argument('--cam',help='detect with input is camera, example: video0 -> 0')
    args = parser.parse_args()
    
    #begin detection coding
    print("cv_detect_inventory running...")
    #load model and get class name
    class_name = read_classes('class.txt')
    OOS_model = load_model_yolov4('yolov4-tiny-OOS.weights','yolov4-tiny-custom-OOS.cfg')
    obj_model = load_model_yolov4('yolov4-tiny-obj.weights','yolov4-tiny-custom-obj.cfg')

    #while loop
    while True:
        if args.cam != None:
            print('start detect with video' + str(args.cam) + "...")
            img = get_image_from_cam(args.cam)
            #if failed
            if img == ERROR:
                break
        
        elif args.img != None:
            print("start detect with " + str(args.img) + "...")
            img = load_image_local("./test/OOS/" + str(args.img))
        else:
            print("can't not detect without input")
            break

        classids_oos, scores_oos, boxes_oos = OOS_model.detect(img, Conf_threshold, NMS_threshold)
        classids_obj, scores_obj, boxes_obj = obj_model.detect(img, Conf_threshold, NMS_threshold)
        change_id_object(classids_obj)

        #append detection beweent object and OOS
        classids = np.append(classids_oos,classids_obj)
        scores = np.append(scores_oos,scores_obj)
        boxes = append_boxes(boxes_oos,boxes_obj)

        
        #draw bouding box in image
        get_detection(class_name,classids,scores,boxes,img)

        #get total of in-stock and OOS
        OOS = len(classids_oos)
        In_stock = len(classids_obj)       
        avalible = "{:.2f}".format(get_area_boxes(boxes_obj)/(get_area_boxes(boxes_obj) + get_area_boxes(boxes_oos)))
        #log total position
        print("total empty position : " + str(OOS))
        print("total obj position : " + str(In_stock))
        print("avalible on shelf: " + str(avalible) + '%')
           
        #save image output
        image_name = 'result.jpg'
        cv.imwrite(image_name,img)

         #post data to firebase
        connect_to_firebase()
        post_to_firebaserealtime(OOS,In_stock,avalible,image_name)
        upload_to_firebaseStorage(image_name)
        break
        #delay_secconds(10)
    #end loop
    print('Stopping cv_detect_inventory.')
    return   

#function test     
def test():
    
    return

if __name__ == "__main__":
    #test()
    main()