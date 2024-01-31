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

'''
datasets = '/home/xyz/workspace/08_dataset/02_depalletize/01_box/all_datasets/project/xiangtan_hairou_20210129'
datasets = '/home/xyz/workspace/01_git/data_generation/xyz-object-detection-models2/test'
datasets = '/home/xyz/workspace/08_dataset/02_depalletize/01_box/all_datasets/project/guoyao_danchai_20210412'
datasets = '/home/xyz/workspace/08_dataset/02_depalletize/01_box/label/huizhou_20211213'
datasets = '/home/xyz/workspace/08_dataset/02_depalletize/01_box/all_datasets/test/mix-depalletize_log_20200610'
datasets_ = '/media/liaolin_data/04_datasets/madai/madai'
datasets_ = '/media/hd4t/liaolin/depalletize'

cropped_dir = '/media/liaolin_data/04_datasets/madai/COCO'
cropped_dir = '/media/liaolin_data/04_datasets/alldata_10times'

datasets_ = '/media/hd4t/liaolin/madai_10times'
cropped_dir = '/media/liaolin_data/04_datasets/madai/madai_10times_cropped_plusBox'

datasets_ = '/media/hd4t/liaolin/depalletize'
cropped_dir = '/media/liaolin_data/04_datasets/box_bag_10time'

datasets_ = '/media/hd4t/liaolin/test/'
cropped_dir = '/media/hd4t/liaolin/test_coco'

datasets_ = '/home/liaolin/04_datasets/bag_filtered'
cropped_dir = '/home/liaolin/04_datasets/bag_filtered_cropped'
datasets_ = '/home/liaolin/04_datasets/bag'
cropped_dir = '/home/liaolin/04_datasets/bag_cropped'

datasets_ = '/home/liaolin/04_datasets/bag_filter_10times'
cropped_dir = '/home/liaolin/04_datasets/bag_filter_10times_coco'

datasets_ = '/media/hd4t/liaolin/box_2times'
cropped_dir = '/media/hd4t/liaolin/box_2times_cropped'

datasets_ = '/home/liaolin/04_datasets/all_datasets/pp_huasheng'
cropped_dir = '/home/liaolin/04_datasets/all_datasets/pp_huasheng_cropped'

datasets_ = '/home/liaolin/04_datasets/all_datasets/pp_liaobao/xyz_format'
cropped_dir = '/home/liaolin/04_datasets/all_datasets/pp_liaobao/cropped'

datasets_= '/home/liaolin/04_datasets/all_datasets/pp_huasheng'
cropped_dir = '/home/liaolin/04_datasets/all_datasets/pp_huasheng_cropped2'

datasets_ = '/home/liaolin/04_datasets/all_datasets/pp_RD22_1116_3/liaobao3/train'
cropped_dir = '/home/liaolin/04_datasets/all_datasets/pp_RD22_1116_3/liaobao3'

datasets_ = '/home/liaolin/04_datasets/all_datasets/pp_RD22_1116_3/liaobao4/train'
cropped_dir = '/home/liaolin/04_datasets/all_datasets/pp_RD22_1116_3/liaobao4'

datasets_ = '/media/hd4t/liaolin/liaobao/test'
cropped_dir = '/media/hd4t/liaolin/liaobao/test_data'

datasets_ = '/media/hd4t/liaolin/liqun_4liaobao/85'
cropped_dir = '/media/hd4t/liaolin/liqun_4liaobao/'

datasets_ = '/home/liaolin/04_datasets/all_datasets/pp_RD22_1116_5/train_70'
cropped_dir = '/home/liaolin/04_datasets/all_datasets/pp_RD22_1116_5'

datasets_ = '/media/ssd4t/liaolin/01_box/box_5times'
cropped_dir = '/media/ssd4t/liaolin/01_box/box_5times_cropped_for_train'

datasets_ = '/media/ssd4t/liaolin/01_box/box'
cropped_dir = '/media/ssd4t/liaolin/01_box/box_cropped_for_test'

datasets_ = '/media/ssd4t/liaolin/02_bag/bag'
cropped_dir = '/media/ssd4t/liaolin/02_bag/bag_cropped_for_test'

datasets_ = '/media/ssd4t/liaolin/02_bag/bag_5times'
cropped_dir = '/media/ssd4t/liaolin/02_bag/bag_5times_cropped_for_train'
'''

datasets_ = '/media/ssd4t/liaolin/01_box/box_5times_v2'
cropped_dir = '/media/ssd4t/liaolin/01_box/box_5times_v2_cropped_for_train'

datasets_ = '/media/ssd4t/liaolin/01_box/box_2times_v3_NoRotation'
cropped_dir = '/media/ssd4t/liaolin/01_box/box_2times_v3_NoRotation_cropped_for_train'

datasets_ ='/media/ssd4t/liaolin/02_bag/bag_10times'
cropped_dir = '/media/ssd4t/liaolin/02_bag/bag_10times_cropped_for_train'

datasets_ = '/media/ssd4t/liaolin/02_bag/bag_10times_Rotate90'
cropped_dir = '/media/ssd4t/liaolin/02_bag/bag_10times_Rotate90_cropped_for_train'

datasets_ = '/media/ssd4t/liaolin/01_box/box_guoyao_10times'
cropped_dir = '/media/ssd4t/liaolin/01_box/box_guoyao_10times_for_yolox'

datasets_ = '/media/ssd4t/liaolin/01_box/box_guoyao_10times_R10degree'
cropped_dir = '/media/ssd4t/liaolin/01_box/box_guoyao_10times_R10degree_for_yolox'

datasets_ = '/media/ssd4t/liaolin/01_box/box_guoyao_3times_R10_Strap'
cropped_dir = '/media/ssd4t/liaolin/01_box/box_guoyao_3times_R10_Strap_for_yolox'

datasets_ = '/media/ssd4t/liaolin/01_box/box_guoyao'
cropped_dir = '/media/ssd4t/liaolin/01_box/box_guoyao_for_yolox'

datasets_ = '/media/ssd4t/liaolin/01_box/box_guoyao'
cropped_dir = '/media/ssd4t/liaolin/01_box/box_guoyao_base_for_yolox'

datasets_ = '/media/ssd4t/liaolin/01_box/box_guoyao_5times_Strap_NoRotation_NoRatio'
cropped_dir = '/media/ssd4t/liaolin/01_box/box_guoyao_5times_Strap_NoRotation_NoRatio_for_yolox'

datasets_ = '/media/ssd4t/liaolin/01_box/box_guoyao_Strap_R10_2times/'
cropped_dir = '/media/ssd4t/liaolin/01_box/box_guoyao_Strap_R10_2times_yolox'

datasets_ = '/media/ssd4t/liaolin/03_piecepicking/all_datasets/'
cropped_dir = '/media/ssd4t/liaolin/03_piecepicking/all_datasets_test_for_yolox'

datasets_ = "/media/ssd4t/liaolin/03_piecepicking/R10_NoRatio_2times"
cropped_dir = "/media/ssd4t/liaolin/03_piecepicking/R10_NoRatio_2times_for_yolox"

datasets_ = '/media/ssd4t/liaolin/01_box/train_Strap_R10_2times/'
cropped_dir = '/media/ssd4t/liaolin/01_box/train_Strap_R10_2times_yolox'

datasets_ = '/media/ssd4t/liaolin/01_box/test_Strap_R10_2times/'
cropped_dir = '/media/ssd4t/liaolin/01_box/test_Strap_R10_2times_yolox'
datasets_ = '/media/ssd4t/liaolin/01_box/dpt_test'
cropped_dir = '/media/ssd4t/liaolin/01_box/dpt_test_yolox'

datasets_ = '/media/ssd4t/liaolin/01_box/chinoh_single/10times_v2'
cropped_dir = '/media/ssd4t/liaolin/01_box/chinoh_single/10times_v2_for_yolox'

datasets_ = '/media/ssd4t/liaolin/01_box/single_image_test/wuxiluke'
cropped_dir = '/media/ssd4t/liaolin/01_box/single_image_test/wuxiluke_coco'

datasets_ = '/media/ssd4t/liaolin/01_box/2parts'
cropped_dir = '/media/ssd4t/liaolin/01_box/2parts_coco'

datasets_ = '/media/ssd4t/liaolin/01_box/projects/market_10times'
cropped_dir = '/media/ssd4t/liaolin/01_box/projects/market_10times_coco'

datasets_ = '/media/ssd4t/liaolin/01_box/projects/6-1_yolox'
cropped_dir = '/media/ssd4t/liaolin/01_box/projects/6-1_yolox_coco'
cropped_root = os.path.join(cropped_dir, 'cropped')
if not os.path.exists(cropped_root):
    os.makedirs(cropped_root)
image_id = 0
anno_id = 0
image_name_id = 9999999
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
            image_name_id -= 1
            rgb_, _, anno_ = sample.strip().split(',')
            image_root = os.path.join(datasets, rgb_)
            
            if os.path.exists(os.path.join(cropped_root, os.path.basename(rgb_))):
                print(rgb_)
                continue
            image_id += 1

            image['file_name'] = str(image_name_id) + '.jpg' #os.path.basename(image_root)
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
            img = cv2.imread(os.path.join(datasets, rgb_))
            img = img[int(miny):int(maxy), int(minx):int(maxx)]
            if img.size == 0:
                continue
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
                annotation['area'] = instance['area']
                annotation['bbox'] = [x1, y1, x2 - x1, y2 - y1]
                
                annotation['segmentation'] = []
                data["annotations"].append(copy.deepcopy(annotation))
            image['width'] = maxx - minx
            image['height'] = maxy - miny
            data["images"].append(copy.deepcopy(image))
            threads1 = threading.Thread(target = save_images, args = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB), os.path.join(cropped_root, image['file_name'])))
            threads1.start()
            #cv2.imwrite(os.path.join(cropped_root, image['file_name']), img)  
    with open(os.path.join(cropped_dir, 'instances_train2017.json'), 'w') as f:
        json.dump(data, f)
    print(os.path.join(cropped_dir, 'instances_train2017.json'))
    break
