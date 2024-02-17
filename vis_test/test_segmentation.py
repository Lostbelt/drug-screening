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
from utils import transform_view, mask_to_coordinates



# M = np.array([[1.765021036058813e+03,0,1.138773062221580e+03],
#               [0,1.771407766368966e+03,8.077989696031810e+02],
#               [0,0,1]])

# D = np.array([-0.4433, 0.1769, 0, 0, 0])

# h, w = (1440, 2560)
# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(M, D, (w,h), 1, (w,h))



video_files_dir = sys.argv[1]



directoires =  glob(f'{video_files_dir}/*.*')


count = 0
model = YOLO('weights/tank_detector.pt')

fish_waights = 'weights/zebrafish_detector.pt'

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_b"
sam = sam_model_registry[MODEL_TYPE](checkpoint='weights/sam_vit_b_01ec64.pth')
sam.to(device=DEVICE)

mask_predictor = SamPredictor(sam)


for path in directoires:
    
    file_name = path.split('/')[-1]

    
    vidcap = cv2.VideoCapture(path)
    
    for _ in range(30 * 4):
        success, image = vidcap.read()
    
    
    # image = cv2.undistort(image, M, D, None, newcameramtx)
    vidcap.release()
    cv2.destroyAllWindows()
    
    res = model.predict(image, conf=0.4, verbose=False)
    ans = mask_to_coordinates(image, sorted(res[0].boxes.xyxy.to('cpu').numpy(), key=lambda box: box[0]), mask_predictor)

    
    for i in ans['new_cors']:
        for j in i:
            image = cv2.circle(image, (j[0], j[1]), radius=0, color=(0, 0, 255), thickness=20)
    
    cv2.imwrite(f'vis_test/{file_name}.jpg', image)
