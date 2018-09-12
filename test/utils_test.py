#coding:utf-8
from os.path import join
from os import listdir
import sys
import numpy as np
import cv2
from tqdm import tqdm

sys.path.append('..')
from Detection.MtcnnDetector import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from core.model import P_Net, R_Net, O_Net
from config import cfg

def GetModelPaths(path=cfg.path_output_models):
    path_pnet = join(path, 'pnet', 'pnet-%s'%cfg.epoch[0])
    path_rnet = join(path, 'rnet', 'rnet-%s'%cfg.epoch[1])
    path_onet = join(path, 'onet', 'onet-%s'%cfg.epoch[2])
    return [path_pnet, path_rnet, path_onet]


def GetDetector(model_path,
                thresh_p=0.5, 
                thresh_r=0.66, 
                thresh_o=0.29, 
                min_face_size=25, 
                scale_factor=0.79,
                nms_pnet = 0.7,
                nms_rnet = 0.6,
                nms_onet = 0.6):
    
    paths = GetModelPaths(model_path)
    thresh=[thresh_p, thresh_r, thresh_o]
    detectors = [None, None, None]
    detectors[0] = FcnDetector(P_Net, paths[0])
    detectors[1] = Detector(R_Net, 24, 1, paths[1])
    detectors[2] = Detector(O_Net, 48, 1, paths[2])
    mtcnn_detector = MtcnnDetector(detectors=detectors, 
                                   min_face_size=min_face_size, 
                                   stride=2, 
                                   threshold=thresh, 
                                   scale_factor=scale_factor,
                                   nms_pnet=nms_pnet,
                                   nms_rnet=nms_rnet,
                                   nms_onet=nms_onet)
    return mtcnn_detector
    
def File2GroundTruthInfo(path_labels):
    gt_info = {}
    with open(path_labels, 'r') as f:
        annotations = f.readlines()
        
    for annotation in annotations:
        content = annotation.strip().split(sep=' ')
        file_id = content[0]
        box = [float(num) for num in content[1:]]
        if file_id not in gt_info.keys():
            gt_info[file_id] = box
        else:
            gt_info[file_id] += box
        
        
    for file_id in gt_info:
        boxes = gt_info[file_id]
        assert len(boxes)%4 == 0
        boxes = np.array(boxes).reshape(-1, 4)
        gt_info[file_id] = boxes
     
    num_gt = len(annotations)
    return gt_info, num_gt

        
def ResizedDetect(detector, img):
    h, w, c = img.shape
    size = max(w, h)
    scale = max(size//500, 1.0)
    dim = (int(w/scale), int(h/scale))
    img_resized = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
    bboxes_pred, _ = detector.detect(img_resized)
    
    if bboxes_pred.shape[0] == 0:
        return bboxes_pred
    else:
        
        bboxes_pred[:,:-1]*=scale
        return bboxes_pred  
    
def GetPredictionInfo(detector, path_imgs):
    def pack(bboxes_pred, file_id):
        predictions = []
        if bboxes_pred.shape[0] == 0:
            return []
        assert bboxes_pred.shape[1] == 5
        for prediction in bboxes_pred:
            bbox = prediction[:-1]
            confidence = prediction[-1]
            result = {'bbox':bbox, 'confidence':confidence, 'file_id':file_id}
            predictions.append(result)
        return predictions
    
    info = []
    # for filename in listdir(path_imgs):
    for filename in tqdm(listdir(path_imgs)):
        path_img = join(path_imgs, filename)
        img = cv2.imread(path_img)
        bboxes_pred = ResizedDetect(detector, img)
        result = pack(bboxes_pred, filename)
        info += result
    return info