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

json_root = '/home/xyz/Scrum'
json_root = './instances_val2017.json'

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

datasets = '/home/xyz/workspace/08_dataset/02_depalletize/01_box/all_datasets/project/xiangtan_hairou_20210129'
datasets = '/home/xyz/workspace/01_git/data_generation/xyz-object-detection-models2/test'
datasets = '/home/xyz/workspace/08_dataset/02_depalletize/01_box/all_datasets/project/guoyao_danchai_20210412'
datasets = '/home/xyz/workspace/08_dataset/02_depalletize/01_box/label/huizhou_20211213'
datasets = '/home/xyz/workspace/08_dataset/02_depalletize/01_box/all_datasets/test/mix-depalletize_log_20200610'
datasets_ = '/home/xyz/workspace/08_dataset/02_depalletize/01_box/all_datasets/yellowBox'
datasets_ = '/media/ssd4t/liaolin/01_box/train_Strap_R10_2times'
save_root = '/media/ssd4t/liaolin/01_box/train_Strap_R10_2times_pd_coco_noBox'
'''
datasets_ = '/media/ssd4t/liaolin/01_box/test_Strap_R10_2times'
save_root = '/media/ssd4t/liaolin/01_box/test_Strap_R10_2times_pd_coco'

datasets_ = '/home/liaolin/temp2'
save_root = '/home/liaolin/temp2_pp_coco'

datasets_ = '/media/ssd4t/liaolin/01_box/temp2'
save_root = '/media/ssd4t/liaolin/01_box/temp2_pd_coco_noBox'
'''
image_id = 0
anno_id = 0
if not os.path.exists(save_root):
    os.makedirs(save_root)

for root, dirs, filenames in os.walk(datasets_):
    for dir in dirs:
        print(dir)
        datasets = os.path.join(root, dir)
        samples = datasets + '/samples.txt'
        s = open(samples, 'r')
        samples_data = s.readlines()
        s.close()
        cropped_root = os.path.join(datasets, 'cropped')
        #if not os.path.exists(cropped_root):
        #    os.makedirs(cropped_root)
        for sample in tqdm(samples_data):
            image_id += 1
            rgb_, _, anno_ = sample.strip().split(',')
            image_root = os.path.join(datasets, rgb_)
            image['file_name'] = os.path.basename(image_root)
            image['id'] = image_id
            
            anno_root = os.path.join(datasets, anno_)
            annos_ = open(anno_root, 'r')
            annos = json.load(annos_)
            instances = annos['instances']
            annotation['image_id'] = image_id
            annotation['category_id'] = 1
            roi = annos['roi']
            roi = np.array(roi)
            minx, miny = roi.min(axis = 0)
            maxx, maxy = roi.max(axis = 0)
            minx, miny, maxx, maxy = int(minx), int(miny), int(maxx), int(maxy)
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
                annotation['area'] = instance['area']
                #annotation['bbox'] = [x1, y1, x2 - x1, y2 - y1]
                
                annotation['segmentation'] = []
                segs = instance['segmentation'][0]
                '''
                for seg in segs:
                    seg[0] = min(maxx, max(0, seg[0] - minx))
                    seg[1] = min(maxx, max(0, seg[1] - miny))
                    annotation['segmentation'].append(seg[0])
                    annotation['segmentation'].append(seg[1])
                '''
                for seg_i in range(len(segs)):
                    seg = segs[seg_i]
                    #print(f"before:{seg}")
                    seg[0] = min(maxx - minx -1, max(0, seg[0] - minx))
                    seg[1] = min(maxy - miny -1, max(0, seg[1] - miny))
                    segs[seg_i] = seg
                    #print(f"after:{seg}")
                #print(f"seg:{segs}")
                center_x = 0
                center_y = 0
                seg_polygon = np.array(segs, dtype=np.int)
                rect = cv2.minAreaRect(seg_polygon)
                box = cv2.boxPoints(rect)
                box = box.astype(float)
                for i in range(4):
                    box[i][0] = min(maxx - minx -1, max(0, box[i][0]))
                    box[i][1] = min(maxy- miny -1, max(0, box[i][1]))
                    annotation['segmentation'].append(box[i][0])
                    annotation['segmentation'].append(box[i][1])
                    
                data["annotations"].append(copy.deepcopy(annotation))
            image['width'] = maxx - minx
            image['height'] = maxy - miny
            data["images"].append(copy.deepcopy(image))
            
            img = cv2.imread(os.path.join(datasets, rgb_))
            img = img[int(miny):int(maxy), int(minx):int(maxx)]
            cv2.imwrite(os.path.join(save_root, os.path.basename(rgb_)), img)
            
    with open(os.path.join(save_root, 'instances_train2017.json'), 'w') as f:
        json.dump(data, f)
    break
    
