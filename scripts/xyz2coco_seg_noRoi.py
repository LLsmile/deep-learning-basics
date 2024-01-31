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
cropped_dir = '/media/ssd4t/liaolin/01_box/yinfang_top_coco_seg'
datasets_ = '/media/ssd4t/liaolin/01_box/projects/yinfang_top_5times'

datasets_ = '/media/ssd4t/liaolin/01_box/projects/jizhuangxiang'
cropped_dir = '/media/ssd4t/liaolin/01_box/projects/jizhuangxiang_coco_seg'

datasets_ = '/media/ssd4t/liaolin/02_bag/bag'
cropped_dir = '/media/ssd4t/liaolin/02_bag/bag_coco_seg'

datasets_ = '/media/ssd4t/liaolin/01_box/projects/yinfang_side_aug'
cropped_dir = '/media/ssd4t/liaolin/01_box/projects/yinfang_side_aug_coco_seg'

datasets_ = '/media/ssd4t/liaolin/01_box/projects/yinfang_top_5times_perspective_aug'
cropped_dir = '/media/ssd4t/liaolin/01_box/projects/yinfang_top_5times_perspective_aug_for_coco_seg'

datasets_ = '/media/ssd4t/liaolin/01_box/projects/yinfang_top_box_5times'
cropped_dir = '/media/ssd4t/liaolin/01_box/projects/yinfang_top_box_5times_for_yolo_seg'

datasets_ = '/media/ssd4t/liaolin/01_box/projects/yinfang_top_box/20231129_5times'

categories = [ {"id": 1, "name": "person"}]
images = []
annotations =[]

datasets_ ='/media/ssd4t/liaolin/01_box/projects/milk'
cropped_dir = '/media/ssd4t/liaolin/01_box/projects/milk_yolo_seg'

datasets_ = '/media/ssd4t/liaolin/01_box/projects/jizhuangxiang_2plane'
cropped_dir = '/media/ssd4t/liaolin/01_box/projects/jizhuangxiang_2plane_for_yolo_seg'

datasets_ = '/media/ssd4t/liaolin/01_box/baokai'
cropped_dir = '/media/ssd4t/liaolin/01_box/baokai_for_yolo_seg'

datasets_ = '/media/ssd4t/liaolin/01_box/projects/6-1_20times'
cropped_dir = '/media/ssd4t/liaolin/01_box/projects/6-1_20times_for_yolo_seg'

datasets_ = '/media/ssd4t/liaolin/01_box/truck_depalletize'
cropped_dir = '/media/ssd4t/liaolin/01_box/truck_depalletize_for_yolo_seg'

datasets_ = '/media/ssd4t/liaolin/01_box/truck_test/2'
cropped_dir = '/media/ssd4t/liaolin/01_box/truck_test/box_maersk_shenlilu_20231215'

datasets_ = '/media/ssd4t/liaolin/01_box/truck_train'

cropped_dir = '/media/ssd4t/liaolin/01_box/truck_train_for_seg'

datasets_ = '/media/ssd4t/liaolin/01_box/projects/PJ23-2421'
cropped_dir = '/media/ssd4t/liaolin/01_box/projects/PJ23-2421_for_seg'

datasets_ = '/media/ssd4t/liaolin/01_box/truck_test/2'
cropped_dir = '/media/ssd4t/liaolin/01_box/truck_test/2_noRoi'

cropped_root = os.path.join(cropped_dir, 'images/train')
if not os.path.exists(cropped_root):
    os.makedirs(cropped_root)
txt_root = os.path.join(cropped_dir, 'labels/train')
if not os.path.exists(txt_root):
    os.makedirs(txt_root)
image_id = 0
anno_id = 0
image_name_id = 1010000
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
            if len(instances) ==0: # or instances[0]['instance_category'] != 'box':
                continue
            annotation['image_id'] = image_id
            annotation['category_id'] = 1
            roi = annos['roi']
            roi = np.array(roi)
            minx, miny = roi.min(axis = 0)
            maxx, maxy = roi.max(axis = 0)
            minx =0
            miny=0
            
            img = cv2.imread(os.path.join(datasets, rgb_))
            print(os.path.join(datasets, rgb_))
            maxx = img.shape[1] - 1
            maxy = img.shape[0] - 1
            img = img[int(miny):int(maxy), int(minx):int(maxx)]
            if img.size == 0:
                continue
            f_txt = open(os.path.join(txt_root, str(image_name_id).zfill(8)+".txt"), 'w')
            line_id = 0
            minx, miny, maxx, maxy = int(minx), int(miny), int(maxx), int(maxy)
            #maxx = img.shape[1] - 1
            #maxy = img.shape[0] - 1
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
                    f_txt.write(' ' + str(seg_x / (maxx - minx)))
                    f_txt.write(' ' + str(seg_y / (maxy -miny)))
                    segmentation.append(seg_x)
                    segmentation.append(seg_y)
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
    '''
    coco_data = {
        "info": {"xyz format to coco format"},
        "licenses": [],
        "categories": categories,
        "images": images,
        "annotations": annotations,
        "type": "instances"
    }
    coco_data = json.dumps(coco_data)
    '''
    data["images"] = images
    data["categories"] = categories
    data["annotations"] = annotations
    print(os.path.join(cropped_dir, 'instances_train2017.json'))
    with open(os.path.join(cropped_dir, 'instances_train2017.json'), 'w') as f:
        json.dump(data, f)
    break
