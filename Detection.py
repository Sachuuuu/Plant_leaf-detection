# -*- coding: utf-8 -*-
"""
Created on Tue May 19 09:21:48 2020

@author: SACHUU
"""

from __future__ import print_function
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image  
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage


class Detect_leaf:
    
    def __init__(self,img_path,leaf_type):
        
        self.img = self.resize_img(cv2.imread(img_path))
        self.orig = self.img.copy()
        self.img = cv2.GaussianBlur(self.img,(7,7),0)
        self.leaf_type = leaf_type
        

    
    def resize_img(self,image):
        
        dims = (800,800)
        resize_imge = cv2.resize(image,dims)
        return resize_imge
    
    def HSV(self):
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (36, 25, 25), (70, 255,255))
        
        mask_1 = mask >0
        Green_img = np.zeros_like(self.img,np.uint8)
        Green_img[mask_1] = self.img[mask_1]
        
        cv2.imwrite("Green_only.png", Green_img)
        plt.imshow(Green_img)
        cv2.waitKey()
        
    def filter_white(self,image, M):
       
        filter_w = np.full((image.shape[0], image.shape[1]), True)
        filter_w[image[:, :, 0] <= 200] = False
        filter_w[image[:, :, 1] <= 220] = False
        filter_w[image[:, :, 2] <= 200] = False
        M[filter_w] = False

        return M



    def filter_black(self,image, M):
       
        filter_b = np.full((image.shape[0], image.shape[1]), True)
        filter_b[image[:, :, 0] >= 50] = False
        filter_b[image[:, :, 1] >= 50] = False   
        filter_b[image[:, :, 2] >= 50] = False
        M[filter_b] = False
        
        return M
    
    def filter_blue(self,image, M):
        
        filter_bl = image[:, :, 0] > image[:, :, 1]
        M[filter_bl] = False
        return M

    def waterShed(self):
        
        image_wshd = cv2.imread("Green_only.png")
        marker = np.full((image_wshd.shape[0], image_wshd.shape[1]), True)
        plt.subplot(2,3,1);plt.imshow(marker)
        marker = self.filter_white(image_wshd,marker)
        plt.subplot(2,3,2);plt.imshow(marker)
        marker = self.filter_black(image_wshd,marker)
        plt.subplot(2,3,3);plt.imshow(marker)
        marker = self.filter_blue(image_wshd,marker)
        plt.subplot(2,3,4);plt.imshow(marker)
        
        new = np.zeros_like(self.orig,np.uint8)
        new[marker] = self.orig[marker]
        plt.imshow(new)
        
        testimg = Image.fromarray(new, 'RGB')
        testimg.save('filtered.png')
            
        
        image_filtered = cv2.imread('filtered.png')
        shifted = cv2.pyrMeanShiftFiltering(image_filtered, 21, 51)
        cv2.imshow("Input", image_filtered)
        cv2.waitKey()
        
        gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        cv2.imshow("Thresh", thresh)
        cv2.waitKey()
        
        ND_image = ndimage.distance_transform_edt(thresh)
        localMax = peak_local_max(ND_image, indices=False, min_distance=20,labels=thresh)
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(-ND_image, markers, mask=thresh)

        

        contours_dict = dict()
        i = 0
        for label in np.unique(labels):
            if label == 0:
                continue
            mask = np.zeros(gray.shape, dtype="uint8")
            mask[labels == label] = 255
            contours, heirarchy   = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                crp = self.img.copy()
                x,y,w,h = cv2.boundingRect(c)
                
                if self.leaf_type == 'Single':
                    area = cv2.contourArea(c)
                    if 10 < area and 10 < w and h > 5:
                        contours_dict[(x, y, w, h)] = c
                        
                else:
                    cv2.rectangle(self.img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    crp = self.orig.copy()[y:y+h, x:x+w]
                    
                    nx,ny,ch = crp.shape
                    
                    if nx >= 100 and ny >=100:
                        cv2.imwrite("cropped/croped_{}.png".format(i),crp)
                        i = i + 1
                        
        if self.leaf_type == "Single":
            img = self.orig.copy()
            contours_filtered = sorted(contours_dict.values(), key=cv2.boundingRect)
            boxes = self.windows(contours_filtered)
            for box in boxes:
                x, y, w, h = box
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imwrite("cropped/croped_{}.png".format(x+y),self.orig.copy()[y:y+h,x:x+w])
            cv2.imshow("cropped",img)
            cv2.waitKey()
            cv2.destroyAllWindows()
        else:
            cv2.imshow("Output", self.img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
    def is_overlapping_horizontally(self,box1, box2):
        x1, _, w1, _ = box1
        x2, _, _, _ = box2
        if x1 > x2:
            return self.is_overlapping_horizontally(box2, box1)
        return (x2 - x1) < w1
    
    
    def merge(self,box1, box2):
        assert self.is_overlapping_horizontally(box1, box2)
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        x = min(x1, x2)
        w = max(x1 + w1, x2 + w2) - x
        y = min(y1, y2)
        h = max(y1 + h1, y2 + h2) - y
        return (x, y, w, h)
    
    
    def windows(self,contours):
        boxes = []
        for cont in contours:
            box = cv2.boundingRect(cont)
            if not boxes:
                boxes.append(box)
            else:
                if self.is_overlapping_horizontally(boxes[-1], box):
                    last_box = boxes.pop()
                    merged_box = self.merge(box, last_box)
                    boxes.append(merged_box)
                else:
                    boxes.append(box)
        return boxes
    
    def main(self):
        self.HSV()
        self.waterShed()


'''
        # Detect_leaf("image path","Single") This is for single leaf image
        
        # Detect_leaf("image path","Multiple") This is for complex/multiple leaves image
        
'''

test = Detect_leaf("Single.jpg","Single")
test.main()