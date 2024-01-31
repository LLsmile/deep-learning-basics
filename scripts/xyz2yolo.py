"""
Copyright (c) XYZ Robotics Inc. - All Rights Reserved.
Unauthorized copying of this file, via any medium is strictly prohibited.
Proprietary and confidential.
Author: lin liao <lin.liao@xyzrobotics.ai>, Nov 2023.
"""
import os
import copy
import json
import cv2
import numpy as np
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", "-d", help="xyz format datasets, split by ,")
    parser.add_argument("--dst", help="yolo datasets")
    parser.add_argument("--type", "-t", help = "convert to detection or segmentation datasets, det for detection and seg for segmentation")
    args = parser.parse_args()

    datasets = args.datasets.split(",")
    dst = args.dst
    yolo_type = args.type
    if not os.path.exists(dst):
        os.makedirs(dst)
    dst_image = os.path.join(dst, "images/train2017")
    if not os.path.exists(dst_image):
        os.makedirs(dst_image)
    dst_labels = os.path.join(dst, "labels/train2017")
    if not os.path.exists(dst_labels):
        os.makedirs(dst_labels)
    train_txt = os.path.join(dst, 'train2017.txt')
    f_list = open(train_txt, 'w')
    for d in datasets:
        print("handling: ", d)
        for root,dirs,files in os.walk(os.path.join(d, "annotation")):
            for file in tqdm(files):
                image_root = os.path.join(os.path.join(d, "data"), file[:-4] + "png")
                if not os.path.exists(image_root):
                    continue
                anno_root = os.path.join(root, file)
                annos = open(anno_root, 'r')
                annos = json.load(annos)
                instances = annos['instances']
                roi = np.array(annos['roi'])
                minx, miny = roi.min(axis = 0)
                maxx, maxy = roi.max(axis = 0)

                img = cv2.imread(image_root)
                minx = min(img.shape[1] - 1, max(0, minx))
                maxx = min(img.shape[1] - 1, max(0, maxx))
                miny = min(img.shape[0] - 1, max(0, miny))
                maxy = min(img.shape[0] - 1, max(0, maxy))
                if maxx <= minx or maxy <= miny:
                    continue
                minx, miny, maxx, maxy = int(minx), int(miny), int(maxx), int(maxy)
                img = img[int(miny):int(maxy), int(minx):int(maxx)]
                if img.shape[0] * img.shape[1] <= 0:
                    continue
                cv2.imwrite(os.path.join(dst_image, os.path.basename(image_root)), img)
                f_list.write("./images/train2017/" + os.path.basename(image_root) + '\n')
                f_label = open(os.path.join(dst_labels, os.path.basename(image_root)[:-4] + '.txt'), 'w')
                roi_w = maxx - minx
                roi_h = maxy - miny
                for instance in instances:
                    if yolo_type == "det":
                        x1,y1,w,h = instance['bbox']
                        x2 = max(0, min(x1 + w - minx, maxx - minx - 1))
                        y2 = max(0, min(y1 + h - miny, maxy - miny - 1))
                        x1 = max(0, x1 - minx)
                        y1 = max(0, y1 - miny)
                        x2 = x1 + w / 2
                        y2 = y1 + h / 2
                        f_label.write("0 ")
                        f_label.write(str(round(x2 /roi_w,6)) + ' ' + str(round(y2 /roi_h,6)) + ' '+str(round(w/roi_w, 6))+' '+str(round(h/roi_h, 6)) + '\n')
                    else:
                        try:
                            segs = instance['parts'][0]['polygon'] # only support one suction area
                        except:
                            print(image_root)
                            continue
                        x1, y1, x2, y2 = maxx, maxy, minx, miny
                        f_label.write('0')
                        for seg in segs:
                            seg_x = max(0, seg[0] - minx)
                            seg_y = max(0, seg[1] - miny)
                            f_label.write(' ' + str(seg_x / (maxx - minx)))
                            f_label.write(' ' + str(seg_y / (maxy -miny))) 
                        f_label.write('\n')
            break
    f_list.close()
if __name__ == '__main__':
    main()