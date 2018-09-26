#coding:utf-8
import numpy as np
import sys
import tabulate as tab
import pandas as pd
from os.path import join

from utils_test import GetDetector, File2GroundTruthInfo, GetPredictionInfo
sys.path.append('..')
from metric.ap import AveragePrecisionOnImages
from Detection.MtcnnDetector import nms_mode
from config import cfg


def MyPrintDF(df):
    table = tab.tabulate(df, headers='keys', tablefmt='psql')
    print(table)
    return

    
class RandomGenerator(object):
    def Random(self):
        raise NotImplementedError
    def View(self):
        raise NotImplementedError
        

# uniform distribution over a range (min, max), value type can be float or integer
class UniformGenerator(RandomGenerator):
    def __init__(self, min, max, type='float'):
        assert type in ['int', 'float']
        self.max = max
        self.min = min
        self.type = type
        
    def Random(self):
        if self.type == 'int':
            return np.random.randint(low=self.min, high=self.max)
        if self.type == 'float':
            result = np.random.uniform(low=self.min, high=self.max)
            return round(result, 3)
            
    def View(self, value):
        return value
        
# the log of random values are uniform distributed over a range (min, max)
class LogGenerator(RandomGenerator):
    def __init__(self, min, max):
        self.max = max
        self.min = min
 
    def Random(self):
        result =  np.exp(np.random.uniform(low=self.min, high=self.max))
        return round(result, 3)
       
    def View(self, value):
        return np.log(value)
            
# uniform distribution over several discrete values
class DiscreteGenerator(RandomGenerator):
    '''
    mapping is a dictionary, for example
    {'Adam':0, 'sgd':1, 'RMSprop':2, 'Adagrad':3}
    '''
    def __init__(self, mapping):
        self.mapping = mapping
        self.keys = list(mapping.keys())
        self.rev = {mapping[key]:key for key in mapping}

    def Random(self):
        idx = np.random.choice(np.arange(len(self.keys)))
        return self.mapping[self.keys[idx]]
       
    def View(self, value):
        return self.rev[value]
        

if __name__=='__main__':
    path_models = 'test\\PreTrainedModels\\HighWay'
    path_models = join(cfg.path_code, path_models)
    path_detection_labels = 'E:\\DM\\MTCNN\\PCD\\Test\\anno_detection.txt'
    path_detection_imgs = 'E:\\DM\\MTCNN\\PCD\\Test\\imgs'
    
    
    iterations = 50 
    min_overlap = 0.7
    tmp_dict = {'Union':1}

    minsize = UniformGenerator(min=25, max=160, type='int')
    factor = UniformGenerator(min=0.7, max=0.9)
    thresh_pred_p = UniformGenerator(min=0.10, max=0.85)
    thresh_pred_r = UniformGenerator(min=0.3, max=0.65)
    thresh_pred_o = UniformGenerator(min=0.05, max=0.7)
    
    thresh_nms_p = UniformGenerator(min=0.55, max=0.8)
    thresh_nms_r = UniformGenerator(min=0.7, max=0.8)
    thresh_nms_o = UniformGenerator(min=0.5, max=0.8)

    mode_p = DiscreteGenerator(mapping=nms_mode)
    mode_r = DiscreteGenerator(mapping=tmp_dict)
    mode_o = DiscreteGenerator(mapping=nms_mode)
    
    thresh_merge = UniformGenerator(min=0.5, max=0.8)
    
    
    mode_merge = DiscreteGenerator(mapping=tmp_dict)
    
   
    results = []
    gt_info, num_gt = File2GroundTruthInfo(path_detection_labels)  
    
    for i in range(iterations):
        
        print('--------------------')
        print('current iteration is {}'.format(i+1))
        print('--------------------')
        kwargs = {}
        kwargs['model_path'] = path_models
        kwargs['minsize'] = minsize.Random()
        kwargs['factor'] = factor.Random()
        
        kwargs['thresh_pred_p'] = thresh_pred_p.Random()
        kwargs['thresh_pred_r'] = thresh_pred_r.Random()
        kwargs['thresh_pred_o'] = thresh_pred_o.Random()
        
        kwargs['thresh_nms_p'] = thresh_nms_p.Random()
        kwargs['thresh_nms_r'] = thresh_nms_r.Random()
        kwargs['thresh_nms_o'] = thresh_nms_o.Random()
        
        kwargs['mode_p'] = mode_p.Random()
        kwargs['mode_r'] = mode_r.Random()
        kwargs['mode_o'] = mode_o.Random()
        
        kwargs['thresh_merge'] = thresh_merge.Random()
        kwargs['mode_merge'] = mode_merge.Random()
        
        
        detector = GetDetector(**kwargs)
        predictions = GetPredictionInfo(detector, path_detection_imgs)  
        ap, _, _ = AveragePrecisionOnImages(gt_info, predictions, num_gt, min_overlap=min_overlap, validate_input=False)
        
        # ap = np.random.random()
        kwargs['ap'] = round(ap, 3)
        del kwargs['model_path']
        kwargs['mode_p'] = mode_p.View(kwargs['mode_p'])
        kwargs['mode_r'] = mode_r.View(kwargs['mode_r'])
        kwargs['mode_o'] = mode_o.View(kwargs['mode_o'])
        kwargs['mode_merge'] = mode_merge.View(kwargs['mode_merge'])
        results.append(kwargs)
        
    d = {key:[dictionary[key] for dictionary in results] for key in results[0].keys()}
    df = pd.DataFrame(d)
    df.sort_values(by='ap', ascending=False, inplace=True)
    df.to_csv('hyperparam.txt', index=None, sep=' ', mode='a')
    MyPrintDF(df)


    