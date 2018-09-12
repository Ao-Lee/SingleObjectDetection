#coding:utf-8
import sys
from os.path import join
from utils_test import GetDetector, File2GroundTruthInfo, GetPredictionInfo
sys.path.append('..')
from metric.ap import AveragePrecisionOnImages
from config import cfg

def Test(path_detection_labels, path_detection_imgs, path_models):
    gt_info, num_gt = File2GroundTruthInfo(path_detection_labels)  
    # if not 'detector' in dir(): detector = GetDetector()
    detector = GetDetector(model_path=path_models)
    predictions = GetPredictionInfo(detector, path_detection_imgs)  
    ap, _, _ = AveragePrecisionOnImages(gt_info, predictions, num_gt, min_overlap=0.5, validate_input=False)
    print(ap)
    return ap
   
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
  
def Compute_AP_CrowdAI():
    path_detection_labels, path_detection_imgs, path_models = GetCrowdAIPath()
    Test(path_detection_labels, path_detection_imgs, path_models)
    
def Compute_AP_HighWay():
    path_detection_labels, path_detection_imgs, path_models = GetHighWayPath()
    Test(path_detection_labels, path_detection_imgs, path_models)
    
if __name__=='__main__':
    Compute_AP_CrowdAI()
    Compute_AP_HighWay()
