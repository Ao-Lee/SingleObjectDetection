#coding:utf-8
import numpy as np
import sys
import tabulate as tab
import pandas as pd

from utils_test import GetDetector, File2GroundTruthInfo, GetPredictionInfo
sys.path.append('..')
from metric.ap import AveragePrecisionOnImages

def MyPrintDF(df):
    table = tab.tabulate(df, headers='keys', tablefmt='psql')
    print(table)
    return

class RandomGenerator(object):
    '''
    uniform distribution if set mod to 'uniform'
    uniform distribution in its log space if set mod to 'exp'
    '''
    def __init__(self, min, max, mod='uniform', type='float'):
        assert mod in ['uniform', 'exp']
        assert type in ['int', 'float']
        self.max = max
        self.min = min
        self.mod = mod
        self.type = type
        
    def GetRandomValue(self):
        if self.type == 'int':
            return np.random.randint(low=self.min, high=self.max)
        if self.mod == 'exp':
            result =  np.exp(np.random.uniform(low=self.min, high=self.max))
            return round(result, 3)
        if self.mod == 'uniform':
            result = (np.random.uniform(low=self.min, high=self.max))
            return round(result, 3)
            
    def GetHyperValue(self, value):
        if self.type == 'int':
            return value
        if self.mod == 'exp':
            return np.log(value)
        if self.mod == 'uniform':
            return value
            

if __name__=='__main__':
    path_detection_labels = 'E:\\DM\\MTCNN\\PCD\\Test\\anno_detection.txt'
    path_detection_imgs = 'E:\\DM\\MTCNN\\PCD\\Test\\imgs'
    
    
    iterations = 200 
    min_overlap = 0.7
    r_thresh_p = RandomGenerator(min=0.05, max=0.95) # threshold for pnet
    r_thresh_r = RandomGenerator(min=0.05, max=0.95) # threshold for rnet
    r_thresh_o = RandomGenerator(min=0.05, max=0.95) # threshold for onet
    r_min_face = RandomGenerator(min=25, max=200, type='int') # min_face_size
    r_scale = RandomGenerator(min=0.4, max=0.9) # scale_factor
    r_nms_p = RandomGenerator(min=0.2, max=0.8) # threshold for nms pnet
    r_nms_r = RandomGenerator(min=0.2, max=0.8) # threshold for nms rnet
    r_nms_o = RandomGenerator(min=0.2, max=0.8) # threshold for nms onet

    results = []
    gt_info, num_gt = File2GroundTruthInfo(path_detection_labels)  
    
    for i in range(iterations):
        
        print('--------------------')
        print('current iteration is {}'.format(i+1))
        print('--------------------')
        kwargs = {}
        kwargs['thresh_p'] = r_thresh_p.GetRandomValue()
        kwargs['thresh_r'] = r_thresh_r.GetRandomValue()
        kwargs['thresh_o'] = r_thresh_o.GetRandomValue()
        kwargs['min_face_size'] = r_min_face.GetRandomValue()
        kwargs['scale_factor'] = r_scale.GetRandomValue()
        kwargs['nms_pnet'] = r_nms_p.GetRandomValue()
        kwargs['nms_rnet'] = r_nms_r.GetRandomValue()
        kwargs['nms_onet'] = r_nms_o.GetRandomValue()
        
        detector = GetDetector(**kwargs)
        predictions = GetPredictionInfo(detector, path_detection_imgs)  
        ap, _, _ = AveragePrecisionOnImages(gt_info, predictions, num_gt, min_overlap=min_overlap, validate_input=False)
        
        # ap = np.random.random()
        kwargs['ap'] = round(ap, 3)
        results.append(kwargs)
        
    d = {key:[dictionary[key] for dictionary in results] for key in results[0].keys()}
    df = pd.DataFrame(d)
    df.sort_values(by='ap', ascending=False, inplace=True)
    df.to_csv('hyperparam.txt', index=None, sep=' ', mode='a')
    MyPrintDF(df)


    