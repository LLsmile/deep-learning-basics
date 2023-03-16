#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 15:40:13 2023

@author: xyz
"""
import os
import cv2
import numpy as np
import json
import copy
from tqdm import tqdm

src_root = '/media/ssd4t/liaolin/04_cylinder/rendered_images_test'

save_root = '/media/ssd4t/liaolin/04_cylinder/rendered_images_test_coco'

src_root = '/media/ssd4t/liaolin/04_cylinder/rendered_images'
save_root = '/media/ssd4t/liaolin/04_cylinder/rendered_images_train'

json_root = './instances_val2017.json'

f_read=open(json_root, 'r')
data = json.load(f_read)
image = copy.deepcopy(data["images"][0])
image['height'] = 960
image['width'] = 1280
data["images"].clear()
annotation_bak = copy.deepcopy(data["annotations"][0])
data["annotations"].clear()
data["categories"][0]['supercategory'] = 'box'
data["categories"][0]['name'] = 'box'
categories = copy.deepcopy(data["categories"][0])
data["categories"].clear()
data["categories"].append(categories)

if not os.path.exists(save_root):
    os.makedirs(save_root)

image_id = 0
anno_id = 0

for root_m, dirs_m, files_m in os.walk(src_root):
    for d_m in tqdm(dirs_m):
        print(d_m)
        for roots,dirs,files in os.walk(os.path.join(root_m, d_m)):
            print(os.path.join(root_m, d_m))
            for d in dirs:
                color_root = os.path.join(roots, d, "color")
                print(color_root)
                colors = []
                for root_c, dir_c, files_c in os.walk(color_root):
                    colors = files_c
                mask_root = os.path.join(roots, d, "mask")
                masks = []
                for root_t, dir_t, files_t in os.walk(mask_root):
                    masks = files_t
                if len(masks) == 0:
                    print(mask_root + " empty")
                    continue
                for color in colors:
                    image_id += 1
                    img = cv2.imread(os.path.join(color_root, color))
                    annos = []
                    boxes = []
                    minx, miny, maxx, maxy = 9999, 9999, 0, 0
                    for mask in masks:
                        if mask.startswith(color[:-4]):
                            m = cv2.imread(os.path.join(mask_root, mask), cv2.IMREAD_GRAYSCALE)
                            ret, thresh = cv2.threshold(m, 127, 255, 0)
                            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                            if len(contours) == 0:
                                continue
                            contour = contours[0]
                            for c in contours[1:]:
                                contour = np.concatenate((contour, c), axis = 0)
                            size = 0
                            for c  in contours:
                                size += len(c)
                            if size!= len(contour):
                                print("error")
                            rect = cv2.minAreaRect(contour)
                            box = cv2.boxPoints(rect)
                            maxxy = np.amax(box, axis = 0)
                            minxy = np.amin(box, axis = 0)
                            maxx = max(maxx, maxxy[0])
                            maxy = max(maxy, maxxy[1])
                            minx = min(minx, minxy[0])
                            miny = min(miny, minxy[1])
                            boxes.append(box)
                            annotation = copy.deepcopy(annotation_bak)
                            annotation['image_id'] = image_id
                            annotation['category_id'] = 1
                            anno_id += 1
                            annotation['id'] = anno_id
                            annotation['area'] = 1000
                            annotation['segmentation'] = []
                            
                            annotation['bbox'] = cv2.boundingRect(contour)
                            annos.append(annotation)
                            '''
                            box = box.astype(float)
                            for b in box:
                                annotation['segmentation'].append(b[0])
                                annotation['segmentation'].append(b[1])
                            data["annotations"].append(annotation)
                            '''
                    #print(boxes)
                    minx, miny, maxx, maxy = int(minx), int(miny), int(maxx), int(maxy)
                    minx = max(0, minx)
                    miny = max(0, miny)
                    maxx = min(img.shape[1], maxx)
                    maxy = min(img.shape[0], maxy)
                    images = img[miny:maxy, minx:maxx]
                    for box_id in range(len(boxes)):
                        box = boxes[box_id].astype(float)
                        anno = annos[box_id]
                        for b in box:
                            anno['segmentation'].append(b[0] - minx)
                            anno['segmentation'].append(b[1] - miny)
                        data["annotations"].append(anno)
                    name = d_m + " " + d + "_" + color
                    cv2.imwrite(os.path.join(save_root, name), images)
                    image['file_name'] = name
                    image['id'] = image_id
                    image['height'] = maxy - miny
                    image['width'] = maxx -minx
                    data["images"].append(copy.deepcopy(image))
            break
    with open(os.path.join(save_root, 'instances_train2017.json'), 'w') as f:
        json.dump(data, f)
    break
