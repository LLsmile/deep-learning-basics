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

'''
datasets_ = '/media/hd4t/liaolin/04_datasets/all_datasets/pp_RD22_1116_5/train_70'
cropped_dir = '/media/hd4t/liaolin/04_datasets/all_datasets/pp_RD22_1116_5/train_yolo'

datasets_ = '/media/hd4t/liaolin/04_datasets/all_datasets/pp_RD22_1116_3/liaobao3/test'
cropped_dir = '/media/hd4t/liaolin/04_datasets/all_datasets/pp_RD22_1116_3/liaobao3/test_yolo'


datasets_ = '/media/hd4t/liaolin/04_datasets/all_datasets/pp_RD22_1116_5/train'
cropped_dir = '/media/hd4t/liaolin/04_datasets/all_datasets/pp_RD22_1116_5/train90_yolo'
datasets_ = '/media/hd4t/liaolin/liaobao/test'
cropped_dir = '/media/hd4t/liaolin/liaobao/test_yolo'

datasets_ = '/media/hd4t/liaolin/liaobao/train'
cropped_dir = '/media/hd4t/liaolin/liaobao/train_yolo'

datasets_ ='/media/ssd4t/liaolin/01_box/wujin_experiments/test'
cropped_dir = '/media/ssd4t/liaolin/01_box/wujin_experiments/test_yolo'

datasets_ ='/media/ssd4t/liaolin/01_box/wujin_experiments/train'
cropped_dir = '/media/ssd4t/liaolin/01_box/wujin_experiments/train_yolo'

datasets_ ='/media/ssd4t/liaolin/01_box/wujin_experiments/train_10times2'
cropped_dir = '/media/ssd4t/liaolin/01_box/wujin_experiments/train_yolo3'

datasets_ ='/media/ssd4t/liaolin/01_box/box_5times_v2'
cropped_dir = '/media/ssd4t/liaolin/01_box/box_5times_v2_yolov7'

datasets_ ='/media/ssd4t/liaolin/01_box/box'
cropped_dir = '/media/ssd4t/liaolin/01_box/box_test_yolov7'

datasets_ = '/media/ssd4t/liaolin/02_bag/bag_10times'
cropped_dir = '/media/ssd4t/liaolin/02_bag/bag_10times_yolov7'

datasets_ = '/media/ssd4t/liaolin/02_bag/bag'
cropped_dir = '/media/ssd4t/liaolin/02_bag/bag_yolov7'
'''
datasets_ ='/media/ssd4t/liaolin/01_box/box_guoyao_Strap_R10_2times'
cropped_dir = '/media/ssd4t/liaolin/01_box/box_guoyao_Strap_R10_2times_yolov7'

datasets_ ='/media/ssd4t/liaolin/01_box/box_guoyao'
cropped_dir = '/media/ssd4t/liaolin/01_box/box_guoyao_base_yolov7'

datasets_ = '/media/ssd4t/liaolin/03_piecepicking/all_datasets'
cropped_dir = '/media/ssd4t/liaolin/03_piecepicking/all_datasets_for_yolov7'

datasets_ = '/media/ssd4t/liaolin/03_piecepicking/R10_NoRatio_2times'
cropped_dir = '/media/ssd4t/liaolin/03_piecepicking/R10_NoRatio_2times_for_yolov7'

datasets_ = '/media/ssd4t/liaolin/01_box/test_Strap_R10_2times'
cropped_dir = '/media/ssd4t/liaolin/01_box/test_Strap_R10_2time_for_yolov7_v2'

datasets_ = '/media/ssd4t/liaolin/01_box/train'
cropped_dir = '/media/ssd4t/liaolin/01_box/train_for_yolov7'

datasets_ = '/media/ssd4t/liaolin/01_box/baokai_Strap_R10_5times'
cropped_dir = '/media/ssd4t/liaolin/01_box/baokai_Strap_R10_5times_for_yoolov7'

datasets_ = '/media/ssd4t/liaolin/01_box/chinoh_train'
cropped_dir = '/media/ssd4t/liaolin/01_box/chinoh_train_for_yolov7'
'''
datasets_ = '/media/ssd4t/liaolin/01_box/baokai_Strap_R10_5times'
cropped_dir = '/media/ssd4t/liaolin/01_box/baokaiNoStrap_OtherStrap_for_yoolov7'

datasets_ = '/media/ssd4t/liaolin/01_box/baokai_small_4times'
cropped_dir = '/media/ssd4t/liaolin/01_box/baokai_small_4times_for_yolov7_test'
'''
datasets_ = '/media/ssd4t/liaolin/01_box/chinoh'
cropped_dir = '/media/ssd4t/liaolin/01_box/chinoh_test_for_yolov7'

datasets_ = '/media/ssd4t/liaolin/01_box/baokai_original_20times'
cropped_dir = '/media/ssd4t/liaolin/01_box/baokai_Strap_R10_5times_for_yoolov7'

datasets_ = '/media/ssd4t/liaolin/01_box/chinoh_single/10times'
cropped_dir = '/media/ssd4t/liaolin/01_box/chinoh_single/10times_for_yolov7'

datasets_ = '/media/ssd4t/liaolin/01_box/box_guoyao'
cropped_dir = '/media/ssd4t/liaolin/01_box/box_guoyao_for_yolov7'

datasets_ = '/media/ssd4t/liaolin/05_temp/money'
datasets_ = '/media/ssd4t/liaolin/05_temp/fake_20231017'
cropped_dir = '/media/ssd4t/liaolin/05_temp/money_yolov7'

datasets_ = '/media/ssd4t/liaolin/01_box/projects/money'
cropped_dir = '/media/ssd4t/liaolin/01_box/projects/money_for_yolov7'

datasets_ = '/media/ssd4t/liaolin/05_temp/money_with_coin_5times'
datasets_ = '/media/ssd4t/liaolin/05_temp/coin'
cropped_dir = '/media/ssd4t/liaolin/05_temp/coin_for_yolov7'

prefix = './images/train2017/'
if not os.path.exists(cropped_dir):
    os.makedirs(cropped_dir)
cropped_root = os.path.join(cropped_dir, 'images')
if not os.path.exists(cropped_root):
    os.makedirs(cropped_root)
label_root = os.path.join(cropped_dir, 'labels')
if not os.path.exists(label_root):
        os.makedirs(label_root)
image_id = 0
anno_id = 0
image_name_id = 9899999
train_txt = os.path.join(cropped_dir, 'train.txt')
f_list = open(train_txt, 'w')
category_dicts = {        
        "coin1": 0,
        "coin5": 1,
        "coin10": 2,
        "unknown": 3}
for root, dirs, filenames in os.walk(datasets_):
    dirs.sort()
    for dir in dirs:
        if 0: #dir != 'HC_20220723_10times':
            continue
        print(dir)
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

            file_name = str(image_name_id)+'.png' 
            #f_list.write(prefix+file_name + '\n')
            anno_root = os.path.join(datasets, anno_)
            annos_ = open(anno_root, 'r')
            annos = json.load(annos_)
            instances = annos['instances']
            roi = annos['roi']
            roi = np.array(roi)
            minx, miny = roi.min(axis = 0)
            maxx, maxy = roi.max(axis = 0)
            img = cv2.imread(os.path.join(datasets, rgb_))
            img = img[int(miny):int(maxy), int(minx):int(maxx)]
            if img.shape[0] ==0:
                continue
            f_list.write(prefix+file_name + '\n')
            f_label = open(os.path.join(label_root, str(image_name_id) + '.txt'), 'w')
            minx, miny, maxx, maxy = int(minx), int(miny), int(maxx), int(maxy)
            roi_w = maxx - minx
            roi_h = maxy - miny
            for instance in instances:
                x1,y1,w,h = instance['bbox']
                x2 = max(0, min(x1 + w - minx, roi_w - 1))
                y2 = max(0, min(y1 + h - miny, roi_h - 1))
                x1 = max(0, x1 - minx)
                y1 = max(0, y1 - miny)
                w = max(0, x2 - x1)
                h = max(0, y2 - y1)
                x2 = x1 + w / 2
                y2 = y1 + h / 2
                if w <= 10 or h <= 10:
                    continue
                #f_label.write('0 ')
                f_label.write(str(category_dicts[instance['instance_category']]) + " ")
                f_label.write(str(round(x2 /roi_w,6)) + ' ' + str(round(y2 /roi_h,6)) + ' '+str(round(w/roi_w, 6))+' '+str(round(h/roi_h, 6)) + '\n')
            f_label.close()
            #print(file_name)
            cv2.imwrite(os.path.join(cropped_root, file_name), img)  
    f_list.close()
    break
