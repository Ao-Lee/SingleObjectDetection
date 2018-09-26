#coding:utf-8
from os.path import join
import numpy as np
import sys
import cv2
from utils_test import GetDetector, GetBestHyperParam_HighWay

sys.path.append('..')
from config import cfg
from utils.viz import PlotImgWithBboxes, ShowImg, MergeImage

class OneImgTestor():
    def __init__(self, path_detection_labels, path_detection_imgs, path_models, hyper_params=None):
        path_detection_labels = join(cfg.path_code, path_detection_labels)
        path_detection_imgs = join(cfg.path_code, path_detection_imgs)
        path_models = join(cfg.path_code, path_models)
        
        if hyper_params is None:
            self.detector = GetDetector(model_path=path_models)
        else:
            self.detector = GetDetector(model_path=path_models, **hyper_params)
            
        self.path_detection_labels = path_detection_labels
        self.path_detection_imgs = path_detection_imgs
        
    def Run(self, size, mod):
        assert mod in ['one', 'multi']

        images = []
        with open(self.path_detection_labels, 'r') as f:
            annotations = f.readlines()
    
        indices = np.random.permutation(np.arange(len(annotations)))
        indices = indices[:size]
    
        for idx in indices:
            content = annotations[idx].strip().split(sep=' ')
            path_img = join(self.path_detection_imgs, content[0])
            img = cv2.imread(path_img)
            bboxes = content[1:]
            bboxes = np.array(bboxes, dtype=np.int).reshape(-1, 4)
            bboxes_pred, landmarks_pred = self.detector.detect(img)
            
            if mod == 'multi': img = PlotImgWithBboxes(img, bboxes, color=(225, 255, 0))
            img = PlotImgWithBboxes(img, bboxes_pred, color=(0, 255, 225))
            img= img[:, :, ::-1]  
            if mod == 'multi': ShowImg(img)
            if mod == 'one': images.append(img) 
            
        if mod == 'one':
            result = MergeImage(images, how='horizontal', color=(70,70,70))
            ShowImg(result)
            
def GetCrowdAITestor():
    path_detection_labels = 'test\\datasets\\TestDatasets\\CrowdAI\\anno_detection.txt'
    path_detection_imgs = 'test\\datasets\\TestDatasets\\CrowdAI\\imgs'
    path_models = 'test\\PreTrainedModels\\CrowdAI'
    testor = OneImgTestor(path_detection_labels, path_detection_imgs, path_models)
    return testor 
        
def GetHighWayTestor():
    path_detection_labels = 'test\\datasets\\TestDatasets\\HighWay\\anno_detection.txt'
    path_detection_imgs = 'test\\datasets\\TestDatasets\\HighWay\\imgs'
    path_models = 'test\\PreTrainedModels\\HighWay'
    kwargs = GetBestHyperParam_HighWay()
    testor = OneImgTestor(path_detection_labels, path_detection_imgs, path_models, hyper_params=kwargs)
    return testor 
    
    
if __name__=='__main__':
    crowdAI_testor = GetCrowdAITestor()
    crowdAI_testor.Run(size=8, mod='multi')
    
    #highway_testor = GetCrowdAITestor()
    #highway_testor.Run(size=5, mod='one')
    
