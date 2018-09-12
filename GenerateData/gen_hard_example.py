#coding:utf-8
import sys
import os
sys.path.insert(0,'..')
import numpy as np
from os.path import join
import pickle
import cv2
from tqdm import tqdm

from core.model import P_Net, R_Net
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from Detection.MtcnnDetector import MtcnnDetector
from config import cfg

try:
    from .loader import TestLoader
    from .utils import IoU, convert_to_square
except ImportError:
    from loader import TestLoader
    from utils import IoU, convert_to_square

def SaveTxt(file, target, idx, content, types):
    assert types in ['part', 'neg', 'pos']
    name = join('%s_%s'%(target, types), '%s.jpg'%idx)
    file.write(name + ' ' + content)
    
def SaveImg(img, target, idx, types):
    folder = join(cfg.path_output_files, '%s_%s'%(target, types))
    path = join(folder, '%s.jpg'%idx)
    cv2.imwrite(path, img)
    

def save_hard_example(target, data, save_path):
    # load ground truth from annotation file
    # format of each line: image/path [x1,y1,x2,y2] for each gt_box in this image
    image_size = cfg.resize[target]
    im_idx_list = data['images']
    gt_boxes_list = data['bboxes']
    num_of_images = len(im_idx_list)

    neg_file = open(join(cfg.path_output_txt, '%s_neg.txt'%target), 'w')
    pos_file = open(join(cfg.path_output_txt, '%s_pos.txt'%target), 'w')
    part_file = open(join(cfg.path_output_txt, '%s_part.txt'%target), 'w')
    
    dirs = ['neg', 'part', 'pos']
    dirs = [join(cfg.path_output_files, '%s_%s'%(target, d)) for d in dirs]
    for d in dirs:
        if not os.path.exists(d): os.makedirs(d)

    det_boxes = pickle.load(open(save_path, 'rb'))

    assert len(det_boxes) == num_of_images, "incorrect detections or ground truths"

    # index of neg, pos and part face, used as their image names
    n_idx = 0
    p_idx = 0
    d_idx = 0
    # image_done = 0
    #im_idx_list image index(list)
    #det_boxes detect result(list)
    #gt_boxes_list gt(list)
    for im_idx, dets, gts in tqdm(zip(im_idx_list, det_boxes, gt_boxes_list), total=num_of_images):
        gts = np.array(gts, dtype=np.float32).reshape(-1, 4)
        
        if dets.shape[0] == 0:
            continue
        img = cv2.imread(im_idx)
        #change to square
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        neg_num = 0
        for box in dets:
            x_left, y_top, x_right, y_bottom, _ = box.astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1

            # ignore box that is too small or beyond image border
            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue

            # compute intersection over union(IoU) between current box and all gt boxes
            Iou = IoU(box, gts)
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv2.resize(cropped_im, (image_size, image_size),
                                    interpolation=cv2.INTER_LINEAR)

            # save negative images and write label
            # Iou with all gts must below 0.3 
            

            if np.max(Iou) < 0.3 and neg_num < 60:
                content = '0\n'
                SaveTxt(neg_file, target, n_idx, content, 'neg')
                SaveImg(resized_im, target, n_idx, 'neg')
                
                n_idx += 1
                neg_num += 1
            else:
                # find gt_box with the highest iou
                idx = np.argmax(Iou)
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt

                # compute bbox reg label
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                # save positive and part-face images and write labels
                if np.max(Iou) >= 0.65:
                    content = '1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2)
                    SaveTxt(pos_file, target, p_idx, content, 'pos')
                    SaveImg(resized_im, target, p_idx, 'pos')
                    p_idx += 1

                elif np.max(Iou) >= 0.4:
                    content = '-1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2)
                    SaveTxt(part_file, target, d_idx, content, 'part')
                    SaveImg(resized_im, target, d_idx, 'part')
                    d_idx += 1
    neg_file.close()
    part_file.close()
    pos_file.close()


def GetModelPaths():
    path_pnet = join(cfg.path_output_models, 'pnet', 'pnet-%s'%cfg.epoch[0])
    path_rnet = join(cfg.path_output_models, 'rnet', 'rnet-%s'%cfg.epoch[1])
    path_onet = join(cfg.path_output_models, 'onet', 'onet-%s'%cfg.epoch[2])
    return [path_pnet, path_rnet, path_onet]

def read_annotation():
    annotation = dict()
    paths = []
    bboxes = []
    with open(cfg.path_detection_labels, 'r') as f:
        file = f.readlines()
    
    for line in file:
        line = line.strip().split(sep=' ')
        path = line[0]
        path = join(cfg.path_detection_imgs, path)
        data = line[1:]
        assert len(data)%4==0
        data = np.array(data, dtype=np.float32).reshape(-1, 4)
        data = list(data)
        data = [list(box) for box in data]
        paths.append(path)
        bboxes.append(data)
        
    annotation['images'] = paths
    annotation['bboxes'] = bboxes
    return annotation
    
    
def Run(target):
    assert target in ['rnet', 'onet']

    model_path = GetModelPaths()
    batch_size = [2048, 256, 16]
    thresh = [0.4, 0.05, 0.7]
    min_face_size = 24

    detectors = [None, None, None]
    detectors[0] = FcnDetector(P_Net, model_path[0])

    if target == 'onet':
        detectors[1] = Detector(R_Net, 24, batch_size[1], model_path[1])

    data = read_annotation()
    
    if cfg.debug:
        data['bboxes'] = data['bboxes'][:20]
        data['images'] = data['images'][:20]

    test_data = TestLoader(data['images'])
    
    mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size, stride=2, threshold=thresh)
    
    detections, _ = mtcnn_detector.detect_face(test_data)
    
    save_file = join(cfg.path_output_files, 'detections_%s.pkl'%target)
    with open(save_file, 'wb') as f:
        pickle.dump(detections, f, 1)
    save_hard_example(target, data, save_file)

def GenHardExampleRnet():
    Run(target='rnet')
def GenHardExampleOnet():
    Run(target='onet')
    
if __name__=='__main__':
    pass
    #GenHardExampleRnet()
    #GenHardExampleOnet()
    
    
    
    