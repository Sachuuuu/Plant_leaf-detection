# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 11:51:30 2020

@author: SACHUU
"""


import cv2
import time
import tensorflow as tf
import tensorflow_hub as hub
import os
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import sklearn
from PIL import Image
import numpy as np

confidence_limit = 0.1
threshold = 0.1


def leaf_detector(img):

    orig = img
    net = cv2.dnn.readNet("yolov3-tiny-obj_final.weights", "yolov3-tiny-obj.cfg")

    np.random.seed(42)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    H, W, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=True)
    net.setInput(blob)

    layerOutputs = net.forward(output_layers)

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > confidence_limit:

                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height), centerX, centerY])
                confidences.append(float(confidence))
                classIDs.append(classID)


    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_limit, threshold)

    crop_boxes = []

    if len(idxs) > 0:

        for i in idxs.flatten():

            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            crop_boxes.append((x, y, x + w, y + h))

            cv2.rectangle(orig, (x, y), (x + w, y + h), (0,255,0), 2)

    return(crop_boxes)
    
def run_model(image_name):

    img = cv2.imread(image_name)
    bounding_boxes = leaf_detector(img)

    i =0
    if (len(bounding_boxes) > 0):
        for box in bounding_boxes:
            crop_img = img[box[1]:box[3], box[0]:box[2]]

            crop_img = cv2.resize(crop_img, (256, 256))
            cv2.imwrite("./detection/cropped-leaf_{}.jpg".format(i), crop_img)
            i = i + 1
