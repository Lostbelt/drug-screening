import numpy as np
import sys
import cv2
from glob import glob
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO
import math
from segment_anything import SamPredictor
import torch
from segment_anything import sam_model_registry
import pandas as pd

from PIL import Image
import time
import matplotlib.patches as patches
import pickle
from scipy.spatial import distance

from utils import transform_view, mask_to_coordinates

M = np.array([[1.765021036058813e+03,0,1.138773062221580e+03],
              [0,1.771407766368966e+03,8.077989696031810e+02],
              [0,0,1]])

D = np.array([-0.4433, 0.1769, 0, 0, 0])

h, w = (1440, 2560)
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(M, D, (w,h), 1, (w,h))




video_files_dir = sys.argv[1]



directoires =  glob(f'{video_files_dir}/*.*)


count = 0
model = YOLO('weights/tank_detector.pt')

fish_waights = 'weights/zebrafish_detector.pt'

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_b"
sam = sam_model_registry[MODEL_TYPE](checkpoint='weights/sam_vit_b_01ec64.pth')
sam.to(device=DEVICE)

mask_predictor = SamPredictor(sam)


data = {}






beg = time.time()



for path in directoires:
    file_name = path.split('/')[-1]
    data[file_name] = []
    start = time.time()
    
    print(path)
    
    vidcap = cv2.VideoCapture(path)
    for _ in range(30 * 4):
        success, image = vidcap.read()
    
    vidcap.release()
    cv2.destroyAllWindows()
    
    # image = cv2.undistort(image, M, D, None, newcameramtx)
    
    res = model.predict(image, conf=0.5, verbose=False)
    ans = mask_to_coordinates(image, sorted(res[0].boxes.xyxy.to('cpu').numpy(), key=lambda box: box[0]), mask_predictor)
    
    if len(ans['new_cors']) == 1:
        mod = 'single'
    else:
        mod = 'duo'
    
    if mod == 'duo':
        trak1 = []
        trak2 = []
        modelv1 = YOLO(fish_waights)
        modelv2 = YOLO(fish_waights)
        
        image_1 = transform_view(image, ans['new_cors'][0])
        h1, w1, _ = image_1.shape
        image_2 = transform_view(image, ans['new_cors'][1])
        h2, w2, _ = image_2.shape
        
        
        
        trak1.append([h1, w1])
        trak2.append([h2, w2])
    
    else:
        trak1 = []

        modelv1 = YOLO(fish_waights)

        
        image = transform_view(image, ans['new_cors'][0])
        h1, w1, _ = image.shape
        trak1.append([h1, w1])
        
    
    
    
    
    vidcap = cv2.VideoCapture(path)
    while True:

        
        success, image = vidcap.read()
        
        if not success:
            break
            
        # image = cv2.undistort(image, M, D, None, newcameramtx)
        
        
        
        
        
        if mod == 'duo':
            left_aq = transform_view(image, ans['new_cors'][0])
            right_aq = transform_view(image, ans['new_cors'][1])
            
            res1 = modelv1.track(left_aq, persist=True, verbose=False, conf=0.5, iou=0.0001)
            res2 = modelv2.track(right_aq, persist=True, verbose=False, conf=0.5, iou=0.0001)
            
            
            trak1.append(res1[0].boxes.data.to('cpu').numpy())
            trak2.append(res2[0].boxes.data.to('cpu').numpy())
            
        
            image1 = res1[0].plot()
            image2 = res2[0].plot()
            
            
            
        else:
            image = transform_view(image, ans['new_cors'][0])

            
            res1 = modelv1.track(image, persist=True, verbose=False, conf=0.5, iou=0.0001)

            
            
            trak1.append(res1[0].boxes.data.to('cpu').numpy())

            
        
            image1 = res1[0].plot()

            
            
    vidcap.release()
    
    if mod == 'duo':
        data[file_name].append(trak1)
        data[file_name].append(trak2)
        
        
    else:
        data[file_name].append(trak1)
    cv2.destroyAllWindows()
    count += 1

    print((time.time() - start) / 60, 'minutes')
    
print(f'It took {(time.time() - beg) / 60 / 60} hours')       
            
            
with open(f'{video_files_dir}.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
