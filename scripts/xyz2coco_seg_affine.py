#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 15:42:10 2021

@author: xyz
"""
import os
import copy
import json
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import threading
import random
import cv2

def rotate_points(points, matrix, image_height, image_width):
    points_ = np.float32(points).reshape([-1,2])
    points_ = np.hstack([points_, np.ones([len(points_), 1])]).T
    target_point = np.dot(matrix, points_)
    target_point = [[target_point[0][x], target_point[1][x]] 
                     for x in range(len(target_point[0]))]
    for index in range(len(target_point)):
        target_point[index][0] = max(0, min(target_point[index][0], image_width -1))
        target_point[index][1] = max(0, min(target_point[index][1], image_height -1))
    return target_point
def save_images(image, name):
    Image.fromarray(image).save(name)

json_root = '/home/xyz/Scrum'
json_root = 'instances_val2017.json'

f_read=open(json_root, 'r')
data = json.load(f_read)
image = copy.deepcopy(data["images"][0])
image['height'] = 960
image['width'] = 1280
data["images"].clear()
annotation = copy.deepcopy(data["annotations"][0])
data["annotations"].clear()
data["categories"][0]['supercategory'] = 'box'
data["categories"][0]['name'] = 'box'
categories = copy.deepcopy(data["categories"][0])
'''
categories['supercategory'] = 'sku'
categories['name'] = 'sku'
'''
data["categories"].clear()
data["categories"].append(categories)

datasets_ = '/media/ssd4t/liaolin/01_box/chinoh/temp'
cropped_dir = '/media/ssd4t/liaolin/01_box/chinoh/temp_coco'

datasets_ = '/media/ssd4t/liaolin/01_box/baokai'
cropped_dir = '/media/ssd4t/liaolin/01_box/baokai_coco'

datasets_ = '/media/ssd4t/liaolin/01_box/test'
cropped_dir = '/media/ssd4t/liaolin/01_box/test_coco'
datasets_ = '/media/ssd4t/liaolin/01_box/chinoh_train'
cropped_dir = '/media/ssd4t/liaolin/01_box/box_all_train_coco'

datasets_ = '/media/ssd4t/liaolin/01_box/baokai_temp'
cropped_dir = '/media/ssd4t/liaolin/01_box/baokai_temp_coco_Seg'

datasets_ = '/media/ssd4t/liaolin/01_box/baokai'
cropped_dir = '/media/ssd4t/liaolin/01_box/baokai_coco_seg'

datasets_ = '/media/ssd4t/liaolin/01_box/projects/jizhuangxiang2_10times'
cropped_dir = '/media/ssd4t/liaolin/01_box/projects/jizhuangxiang_coco_seg2'

datasets_ = '/media/ssd4t/liaolin/01_box/baokai'
cropped_dir = '/media/ssd4t/liaolin/01_box/all_box_coco_seg'

datasets_ = '/media/ssd4t/liaolin/01_box/projects/milk'
cropped_dir = '/media/ssd4t/liaolin/01_box/projects/milk_seg'
datasets_ = '/media/ssd4t/liaolin/01_box/projects/milk_10times'

datasets_ = '/media/ssd4t/liaolin/01_box/temp'
cropped_dir = '/media/xyz/3C0CEC490CEBFFAE/01_datasets/03_depalletize/temp_affine'
datasets_ = '/media/xyz/3C0CEC490CEBFFAE/01_datasets/03_depalletize/temp'

datasets_ = '/media/ssd4t/liaolin/01_box/baokai'
cropped_dir = '/media/ssd4t/liaolin/01_box/baokai_affine'

categories = [ {"id": 1, "name": "person"}]
images = []
annotations =[]

cropped_root = os.path.join(cropped_dir, 'images/train')
if not os.path.exists(cropped_root):
    os.makedirs(cropped_root)
txt_root = os.path.join(cropped_dir, 'labels/train')
if not os.path.exists(txt_root):
    os.makedirs(txt_root)
image_id = 0
anno_id = 0
image_name_id = 0
for root, dirs, filenames in os.walk(datasets_):
    for dir in dirs:
        print(dir)
        if 0: #dir !=  "cemat_2022_mix_dpt_gray": #"abb_test_20211029" and dir != "cemat_dpt_20211020":
            continue
        datasets = os.path.join(root, dir)
        samples = datasets + '/samples.txt'
        s = open(samples, 'r')
        samples_data = s.readlines()
        s.close()

        for sample in tqdm(samples_data):
            image_name_id += 1
            rgb_, _, anno_ = sample.strip().split(',')
            image_root = os.path.join(datasets, rgb_)
            
            if os.path.exists(os.path.join(cropped_root, os.path.basename(rgb_))):
                print(rgb_)
                continue
            image_id += 1

            image['file_name'] = str(image_name_id).zfill(8) + '.png' #os.path.basename(image_root)
            image['id'] = image_id
            
            anno_root = os.path.join(datasets, anno_)
            annos_ = open(anno_root, 'r')
            annos = json.load(annos_)
            instances = annos['instances']
            if len(instances) ==0 or instances[0]['instance_category'] != 'box':
                continue
            annotation['image_id'] = image_id
            annotation['category_id'] = 1
            roi = annos['roi']
            roi = np.array(roi)
            minx, miny = roi.min(axis = 0)
            maxx, maxy = roi.max(axis = 0)
            img = cv2.imread(os.path.join(datasets, rgb_))
            img = img[int(miny):int(maxy), int(minx):int(maxx)]
            if 1:
                new_x = min(maxx - minx - 1, random.randint(5, 20) / 100 * (maxx - minx))
                new_y = min(maxy - miny - 1, random.randint(5, 20) / 100 * (maxy - miny))
                new_x2 = min(maxx - minx - 1, maxx - minx - 1 - random.randint(5, 10) / 100 * (maxx - minx))
                new_y2 = min(maxy - miny - 1, random.randint(1, 5) / 100 * (maxy - miny))
                tar_matrix = [[new_x, new_y], [new_x2, new_y2], [maxx - minx - 1, maxy - miny - 1], [0, maxy - miny - 1]]
                src_matrix = [[0, 0], [maxx - minx - 1, 0], [maxx - minx - 1, maxy - miny - 1], [0, maxy - miny - 1]]
            rotate_matrix = cv2.getPerspectiveTransform(np.float32(src_matrix), np.float32(tar_matrix)) 
            if img.size == 0:
                continue
            img = cv2.warpPerspective(img, rotate_matrix, (int(maxx - minx + 1), int(maxy - miny + 1)))
            cv2.imwrite(os.path.join(cropped_root, image['file_name']), img)
            f_txt = open(os.path.join(txt_root, str(image_name_id).zfill(8)+".txt"), 'w')
            line_id = 0
            minx, miny, maxx, maxy = int(minx), int(miny), int(maxx), int(maxy)
            for instance in instances:
                x1,y1,w,h = instance['bbox']
                x2 = x1 + w
                y2 = y1 + h
            for instance in instances:
                x1,y1,w,h = instance['bbox']
                x2 = x1 + w
                y2 = y1 + h
                x1 = min(maxx - minx - 1, max(0, x1 - minx))
                x2 = min(maxx - minx - 1, max(0, x2 - minx))
                y1 = min(maxy - miny - 1, max(0, y1 - miny))
                y2 = min(maxy - miny - 1, max(0, y2 - miny))
                if (y2 - y1) * (x2 - x1) < w * h / 3:
                    continue
                anno_id += 1
                annotation['id'] = anno_id
                annotation['area'] = (y2 - y1) * (x2 - x1)
                annotation['bbox'] = [x1, y1, x2 - x1, y2 - y1]
                annotation['iscrowd'] = 0
                annotation['segmentation'] = []
                segmentation = []
                try:
                    segs = instance['parts'][0]['polygon']
                except:
                    print(rgb_)
                    continue
                x1, y1, x2, y2 = maxx, maxy, minx, miny
                if line_id:
                    f_txt.write('\n')
                f_txt.write('0')
                line_id += 1
                for seg in segs:
                    seg_x = max(0, seg[0] - minx)
                    seg_y = max(0, seg[1] - miny)
                    segmentation.append(seg_x)
                    segmentation.append(seg_y)
                transformed_points = cv2.perspectiveTransform(np.array(segmentation, dtype= np.float32).reshape(-1, 1, 2), rotate_matrix)
                for transformed_point in transformed_points:
                    f_txt.write(' ' + str(transformed_point[0][0]/ (maxx - minx)))
                    f_txt.write(' ' + str(transformed_point[0][1]/ (maxy -miny)))
                annotation['segmentation'].append(segmentation)

                annotations.append(copy.deepcopy(annotation))
            f_txt.close()
            cv2.imwrite(os.path.join(cropped_root, image['file_name']), img)
            image_info = {
                "id": image_id,  # Unique image ID
                "width": maxx - minx + 1,
                "height": maxy - miny + 1,
                "file_name": image['file_name'],
                "license": None,
                "flickr_url": None,
                "coco_url": None,
                "date_captured": None,
            }
            images.append(copy.deepcopy(image_info))
    data["images"] = images
    data["categories"] = categories
    data["annotations"] = annotations
    # print(os.path.join(cropped_dir, 'instances_train2017.json'))
    # with open(os.path.join(cropped_dir, 'instances_train2017.json'), 'w') as f:
    #     json.dump(data, f)
    break
