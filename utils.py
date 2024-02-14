import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO
import math
from skimage.transform import resize
import matplotlib.patches as patches
from scipy.spatial import distance



def farthest_point(array, x, y):
    '''
    Finds the farthest point from x and y on the mask
    '''
    
    i, j = np.where(array == True)
    coordinates = []
    for one, two in zip(i, j):
        coordinates.append([one, two])
    coordinates = np.array(coordinates)
    max_dist = 0
    for i in coordinates:
        distance = math.dist(i, [x, y])
        if max_dist < distance:
            max_dist = distance
            y1, x1 = i
    return x1, y1


def mask_to_coordinates(image, array, mask_predictor):
    '''
    Takes the tank mask and gives the coordinates of 4 corners and the mask
    '''

    answer = {
    'orig_mask': [],
    'box_cors': [],
    'use_mask': [],
    'new_cors': []
    }

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask_predictor.set_image(image_rgb)
    mask_list = []
    for i in array:
        masks, scores, logits = mask_predictor.predict(
        box=i,
        multimask_output=True)
        mask_list.append(masks[0])
        answer['orig_mask'].append(masks)
        answer['box_cors'].append(i)
        answer['use_mask'].append(masks[0])


    for m, c in zip(mask_list, array):

        x1, y1, x2, y2 = [int(i) for i in c]
        im = m[y1:y2, x1:x2]
        new_im = resize(im, (im.shape[0], int(im.shape[1] * 0.7)))
        x, y = [int(new_im.shape[1]/2), int(new_im.shape[0]/2)]

        abl_l_v = new_im[:y,:x]
        abl_r_v = new_im[:y,x:]
        abl_l_n = new_im[y:,:x]
        abl_r_n = new_im[y:,x:]

        X1, Y1 = farthest_point(abl_l_v, y, x)
        X2, Y2 = farthest_point(abl_r_v, y, 0)
        X3, Y3 = farthest_point(abl_r_n, 0, 0)
        X4, Y4 = farthest_point(abl_l_n, 0, x)

        angles_list = np.array([
        [X1 / 0.7, Y1],
        [(X2 + x) / 0.7, Y2],
        [(X3 + x) / 0.7, Y3 + y],
        [X4 / 0.7, Y4 + y]
        ], dtype='int32') + np.array([x1, y1], dtype='int32')
        
        answer['new_cors'].append(angles_list)
    
    return answer


def transform_view(image, points_list):
    '''
    Transforming an image based on 4 corners
    '''


    x1, y1 = points_list[0]  # Top-left
    x2, y2 = points_list[1]  # Top-right
    x3, y3 = points_list[2]  # Bottom-right
    x4, y4 = points_list[3]  # Bottom-left

    # Calculate the width and height of the new view
    width = max(abs(x2 - x1), abs(x3 - x4))
    height = max(abs(y3 - y2), abs(y4 - y1))


    # Define the new coordinates for the view
    new_coordinates = [[0, 0], [width, 0], [width, height], [0, height]]

    # Calculate the perspective transformation matrix
    perspective_matrix = cv2.getPerspectiveTransform(
        np.float32(points_list), np.float32(new_coordinates)
    )

    # Apply the perspective transformation to crop and align the view
    cropped_image = cv2.warpPerspective(image, perspective_matrix, (width, height))
    return cropped_image


def get_iou(a, b, epsilon=1e-5):
    """ Given two boxes `a` and `b` defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union score for these two boxes.

    Args:
        a:          (list of 4 numbers) [x1,y1,x2,y2]
        b:          (list of 4 numbers) [x1,y1,x2,y2]
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (float) The Intersect of Union score.
    """
    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined+epsilon)
    return iou


def video_to_tank(coordinates, h, w):
    '''
    Converts coordinates from video dimension to tank dimension
    '''
    
    x1, y1, x2, y2 = coordinates
    x, y  = x1/w + ((x2/w - x1/w)/2), y1/h + ((y2/h - y1/h)/2)
    return x*20, y*20



def generate_intermediate_bbox(bbox1, bbox2):
    '''
    Generates an intermediate bounding box
    '''
    
    x1 = (bbox1[0] + bbox2[0]) / 2
    y1 = (bbox1[1] + bbox2[1]) / 2
    x2 = (bbox1[2] + bbox2[2]) / 2
    y2 = (bbox1[3] + bbox2[3]) / 2
    intermediate_bbox = (x1, y1, x2, y2)
    return intermediate_bbox


def visualize_bboxes(image, bboxes):
    # Create figure and axes
    fig, ax = plt.subplots(1)
    
    # Display the image
    ax.imshow(image)
    
    # Add bounding boxes to the image
    for bbox in bboxes:
        # Extract coordinates
        x1, y1, x2, y2 = bbox[:4]
        
        # Calculate width and height
        width = x2 - x1
        height = y2 - y1
        
        # Create a rectangle patch
        rect = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor='r', facecolor='none')
        
        # Add the rectangle to the axes
        ax.add_patch(rect)
    
    # Show the plot
    plt.show()
    
    
def post_processing(raw_data, stop=5*60*30):
    '''
    From the list of predictions throughout the video, finds the initial appearance of the
    fish and cuts out 5 minutes for subsequent processing.
    '''
    
    start = 0
    mod = 'start search'
    clean_data = []
    for i, v in enumerate(raw_data):
        if mod == 'start search':
            if len(v) != 0:
                try:
                    ids = [int(j[:,4]) for j in raw_data[i:i+20]]
                    if len(set(ids)) == 1:
                        start = i
                        mod = 'analyze'
                        clean_data.append(v[0][:4].tolist())
                except:
                    continue
            else:
                continue

        else:

            if len(v) > 1:
                best = None
                best_score = 0
                for n in v:
                    if get_iou(raw_data[i-1][0][:4], n[:4]) > best_score:
                        best = n[:4]
                clean_data.append(best.tolist())
                continue
            if len(v) == 0:
                clean_data.append([])
                continue
            clean_data.append(v[0][:4].tolist())



    while [] in clean_data:
        for i, v in enumerate(clean_data):
            if v == []:
                count = 0
                while True:
                    if clean_data[i+count] == []:
                        count += 1
                    else:
                        break
                i_box = generate_intermediate_bbox(clean_data[i-1], clean_data[i+count])
                clean_data[i+int(count/2)] = i_box
                break
    clean_data = clean_data[:stop]
    return clean_data, start



def xyxy_to_yolo(bbox):
    """
    Convert bounding box from xyxy format to YOLO xywh format.

    Args:
        bbox (tuple): A tuple representing the xyxy bounding box (x_min, y_min, x_max, y_max).
        image_width (int): Width of the image containing the bounding box.
        image_height (int): Height of the image containing the bounding box.

    Returns:
        tuple: A tuple representing the YOLO xywh bounding box (x_center, y_center, width, height).
    """
    x_min, y_min, x_max, y_max = bbox

    # Calculate center coordinates
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0

    # Calculate width and height
    width = x_max - x_min
    height = y_max - y_min
    return x_center, y_center, width, height


def calculate_overlap(box1, box2):
    """
    Calculates how closely two bounding boxes match
    """
    x_center1, y_center1, width1, height1 = xyxy_to_yolo(box1)
    x_center2, y_center2, width2, height2 = xyxy_to_yolo(box2)

    # Calculate coordinates of top-left and bottom-right corners of each box
    x1, y1 = x_center2 - width1 / 2, y_center2 - height1 / 2
    x2, y2 = x_center2 + width1 / 2, y_center2 + height1 / 2
    x3, y3 = x_center2 - width2 / 2, y_center2 - height2 / 2
    x4, y4 = x_center2 + width2 / 2, y_center2 + height2 / 2

    # Calculate intersection area
    x_intersection = max(0, min(x2, x4) - max(x1, x3))
    y_intersection = max(0, min(y2, y4) - max(y1, y3))
    intersection_area = x_intersection * y_intersection

    # Calculate union area
    box1_area = width1 * height1
    box2_area = width2 * height2
    union_area = box1_area + box2_area - intersection_area

    # Calculate overlap
    overlap = intersection_area / union_area

    return overlap
