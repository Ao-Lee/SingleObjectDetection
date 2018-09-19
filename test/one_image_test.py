#coding:utf-8
from os.path import join
import numpy as np
import sys
import cv2
from utils_test import GetDetector

sys.path.append('..')
from config import cfg
from utils.viz import PlotImgWithBboxes, ShowImg, MergeImage

def Test(path_detection_labels, path_detection_imgs, path_models, mod='one', size=4):
    assert mod in ['one', 'multi']
    # if not 'detector' in dir(): detector = GetDetector(model_path=path_models)
    detector = GetDetector(model_path=path_models)
    images = []
    with open(path_detection_labels, 'r') as f:
        annotations = f.readlines()

    indices = np.random.permutation(np.arange(len(annotations)))
    indices = indices[:size]

    for idx in indices:
        content = annotations[idx].strip().split(sep=' ')
        path_img = join(path_detection_imgs, content[0])
        img = cv2.imread(path_img)
        bboxes = content[1:]
        bboxes = np.array(bboxes, dtype=np.int).reshape(-1, 4)
        bboxes_pred, landmarks_pred = detector.detect(img)
        
        if mod == 'multi': img = PlotImgWithBboxes(img, bboxes, color=(225, 255, 0))
        img = PlotImgWithBboxes(img, bboxes_pred, color=(0, 255, 225))
        img= img[:, :, ::-1]  
        if mod == 'multi': ShowImg(img)
        if mod == 'one': images.append(img) 
        
    if mod == 'one':
        result = MergeImage(images, how='horizontal', color=(70,70,70))
        ShowImg(result)

def GetCrowdAIPath():
    path_detection_labels = 'test\\datasets\\TestDatasets\\CrowdAI\\anno_detection.txt'
    path_detection_imgs = 'test\\datasets\\TestDatasets\\CrowdAI\\imgs'
    path_models = 'test\\PreTrainedModels\\CrowdAI'
    
    path_detection_labels = join(cfg.path_code, path_detection_labels)
    path_detection_imgs = join(cfg.path_code, path_detection_imgs)
    path_models = join(cfg.path_code, path_models)
    return path_detection_labels, path_detection_imgs, path_models
    
def GetHighWayPath():
    path_detection_labels = 'test\\datasets\\TestDatasets\\HighWay\\anno_detection.txt'
    path_detection_imgs = 'test\\datasets\\TestDatasets\\HighWay\\imgs'
    path_models = 'test\\PreTrainedModels\\HighWay'
    
    path_detection_labels = join(cfg.path_code, path_detection_labels)
    path_detection_imgs = join(cfg.path_code, path_detection_imgs)
    path_models = join(cfg.path_code, path_models)
    return path_detection_labels, path_detection_imgs, path_models


def Test_CrowdAI():
    path_detection_labels, path_detection_imgs, path_models = GetCrowdAIPath()
    Test(path_detection_labels, path_detection_imgs, path_models, mod='one')
    
def Test_HighWay():
    path_detection_labels, path_detection_imgs, path_models = GetHighWayPath()
    Test(path_detection_labels, path_detection_imgs, path_models, mod='multi')
    

if __name__=='__main__':
    #Test_CrowdAI()
    #Test_HighWay()
    path_detection_labels, path_detection_imgs, path_models = GetCrowdAIPath()
    
    detector = GetDetector(model_path=path_models)
    images = []
    with open(path_detection_labels, 'r') as f:
        annotations = f.readlines()

    content = annotations[0].strip().split(sep=' ')
    path_img = join(path_detection_imgs, content[0])
    img = cv2.imread(path_img)
    bboxes = content[1:]
    bboxes = np.array(bboxes, dtype=np.int).reshape(-1, 4)
    total_boxes, points = detector.detect(img)
    

