#coding:utf-8
import numpy as np
import cv2
import os
from os.path import join
from tqdm import tqdm
import numpy.random as npr

try:
    from .utils import IoU
except ImportError:
    from utils import IoU

import sys
sys.path.append('..')
from config import cfg


def GenExamplePnet(target='pnet'):
    pos_save_dir = join(cfg.path_output_files, target + '_pos')
    part_save_dir = join(cfg.path_output_files, target + '_part')
    neg_save_dir = join(cfg.path_output_files, target + '_neg')
    if not os.path.exists(pos_save_dir):
        os.mkdir(pos_save_dir)
    if not os.path.exists(part_save_dir):
        os.mkdir(part_save_dir)
    if not os.path.exists(neg_save_dir):
        os.mkdir(neg_save_dir)
    
    f1 = open(join(cfg.path_output_txt, target + '_pos.txt'), 'w')
    f2 = open(join(cfg.path_output_txt, target + '_neg.txt'), 'w')
    f3 = open(join(cfg.path_output_txt, target + '_part.txt'), 'w')
    with open(cfg.path_detection_labels, 'r') as f:
        annotations = f.readlines()
    
    if cfg.debug:
        annotations = annotations[:300]
    p_idx = 0 # positive
    n_idx = 0 # negative
    d_idx = 0 # dont care
    idx = 0
    box_idx = 0
    for annotation in tqdm(annotations):
        annotation = annotation.strip().split(' ')
        #image path
        im_path = annotation[0]
        #boxed change to float type
        boxes = annotation[1:]
        #gt
        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        #load image
        img = cv2.imread(os.path.join(cfg.path_detection_imgs, im_path))
        print(os.path.join(cfg.path_detection_imgs, im_path))
        idx += 1
        height, width, channel = img.shape
    
        neg_num = 0
        #1---->50
        while neg_num < 50:
            #neg_num's size [40,min(width, height) / 2],min_size:40 
            size = npr.randint(12, min(width, height) / 2)
            #top_left
            nx = npr.randint(0, width - size)
            ny = npr.randint(0, height - size)
            #random crop
            crop_box = np.array([nx, ny, nx + size, ny + size])
            #cal iou
            Iou = IoU(crop_box, boxes)
            
            cropped_im = img[ny : ny + size, nx : nx + size, :]
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
    
            if np.max(Iou) < 0.3:
                # Iou with all gts must below 0.3
                save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
                name = join('%s_neg'%target, '%s.jpg'%n_idx)
                f2.write(name + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1
        #as for 正 part样本
        for box in boxes:
            # box (x_left, y_top, x_right, y_bottom)
            x1, y1, x2, y2 = box
            #gt's width
            w = x2 - x1 + 1
            #gt's height
            h = y2 - y1 + 1
    
            # ignore small faces
            # in case the ground truth boxes of small faces are not accurate
            if max(w, h) < 40 or x1 < 0 or y1 < 0:
                continue
            for i in range(5):
                size = npr.randint(12, min(width, height) / 2)
                # delta_x and delta_y are offsets of (x1, y1)
                delta_x = npr.randint(max(-size, -x1), w)
                delta_y = npr.randint(max(-size, -y1), h)
                nx1 = int(max(0, x1 + delta_x))
                ny1 = int(max(0, y1 + delta_y))
                if nx1 + size > width or ny1 + size > height:
                    continue
                crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
                Iou = IoU(crop_box, boxes)
        
                cropped_im = img[ny1: ny1 + size, nx1: nx1 + size, :]
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
        
                if np.max(Iou) < 0.3:
                    # Iou with all gts must below 0.3
                    save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                    name = join('%s_neg'%target, '%s.jpg'%n_idx)
                    f2.write(name + ' 0\n')
                    cv2.imwrite(save_file, resized_im)
                    n_idx += 1        
            # generate positive examples and part faces
            for i in range(20):
                # pos and part face size [minsize*0.8,maxsize*1.25]
                size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
    
                # delta here is the offset of box center
                delta_x = npr.randint(-w * 0.2, w * 0.2)
                delta_y = npr.randint(-h * 0.2, h * 0.2)
                #show this way: nx1 = max(x1+w/2-size/2+delta_x)
                nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
                #show this way: ny1 = max(y1+h/2-size/2+delta_y)
                ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
                nx2 = nx1 + size
                ny2 = ny1 + size
    
                if nx2 > width or ny2 > height:
                    continue 
                crop_box = np.array([nx1, ny1, nx2, ny2])
                #yu gt de offset
                offset_x1 = (x1 - nx1) / float(size)
                offset_y1 = (y1 - ny1) / float(size)
                offset_x2 = (x2 - nx2) / float(size)
                offset_y2 = (y2 - ny2) / float(size)
                #crop
                cropped_im = img[ny1 : ny2, nx1 : nx2, :]
                #resize
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
    
                box_ = box.reshape(1, -1)
                if IoU(crop_box, box_) >= 0.65:
                    save_file = os.path.join(pos_save_dir, "%s.jpg"%p_idx)
                    name = join('%s_pos'%target, '%s.jpg'%p_idx)
                    f1.write(name + ' 1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1
                elif IoU(crop_box, box_) >= 0.4:
                    save_file = os.path.join(part_save_dir, "%s.jpg"%d_idx)
                    name = join('%s_part'%target, '%s.jpg'%d_idx)
                    f3.write(name + ' -1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
            box_idx += 1
    f1.close()
    f2.close()
    f3.close()
    
if __name__=='__main__':
    pass
    # GenExamplePnet()
