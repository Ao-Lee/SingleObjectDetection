#coding:utf-8
import sys
from os.path import join
from utils_test import GetDetector, File2GroundTruthInfo, GetPredictionInfo, GetBestHyperParam_HighWay
sys.path.append('..')
from metric.ap import AveragePrecisionOnImages
from config import cfg

class APTestor():
    def __init__(self, path_detection_labels, path_detection_imgs, path_models, min_overlap=0.5, hyper_params=None):
        path_detection_labels = join(cfg.path_code, path_detection_labels)
        path_detection_imgs = join(cfg.path_code, path_detection_imgs)
        path_models = join(cfg.path_code, path_models)
        
        gt_info, num_gt = File2GroundTruthInfo(path_detection_labels)
        
        if hyper_params is None:
            self.detector = GetDetector(model_path=path_models)
        else:
            self.detector = GetDetector(model_path=path_models, **hyper_params)
            
        self.gt_info = gt_info
        self.num_gt = num_gt
        self.path_detection_imgs = path_detection_imgs
        self.min_overlap = min_overlap
        
    def Run(self):
        predictions = GetPredictionInfo(self.detector, self.path_detection_imgs) 
        ap, _, _ = AveragePrecisionOnImages(self.gt_info, predictions, self.num_gt, min_overlap=self.min_overlap, validate_input=False)
        print(ap)
        return ap
        
def GetCrowdAITestor():
    path_detection_labels = 'test\\datasets\\TestDatasets\\CrowdAI\\anno_detection.txt'
    path_detection_imgs = 'test\\datasets\\TestDatasets\\CrowdAI\\imgs'
    path_models = 'test\\PreTrainedModels\\CrowdAI'
    testor = APTestor(path_detection_labels, path_detection_imgs, path_models)
    return testor
   
def GetHighWayTestor():
    path_detection_labels = 'test\\datasets\\TestDatasets\\HighWay\\anno_detection.txt'
    path_detection_imgs = 'test\\datasets\\TestDatasets\\HighWay\\imgs'
    path_models = 'test\\PreTrainedModels\\HighWay'
    kwargs = GetBestHyperParam_HighWay()
    testor = APTestor(path_detection_labels, path_detection_imgs, path_models, hyper_params=kwargs)
    return testor
    
    
def GetTestingTestor():
    path_detection_labels = 'E:\\DM\\MTCNN\\PCD\\Test\\anno_detection.txt'
    path_detection_imgs = 'E:\\DM\\MTCNN\\PCD\\Test\\imgs'
    path_models = 'test\\PreTrainedModels\\HighWay'
    testor = APTestor(path_detection_labels, path_detection_imgs, path_models)
    return testor
    
    
if __name__=='__main__':
    #crowdAI_testor = GetCrowdAITestor()
    #crowdAI_testor.Run()
    
    #highway_testor = GetHighWayTestor()
    #highway_testor.Run()
    
    testor = GetTestingTestor()
    testor.Run()
