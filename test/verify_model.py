#coding:utf-8
from os.path import join
import tensorflow as tf
import numpy as np
import sys
from sklearn import metrics
from utils_test import GetModelPaths

sys.path.append('..')
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from core.model import P_Net, R_Net, O_Net
from config import cfg

def GetModel(target):
    assert target in ['pnet', 'rnet', 'onet']
    paths = GetModelPaths()
    if target == 'pnet':
        return FcnDetector(P_Net, paths[0])
    if target == 'rnet':
        return Detector(R_Net, 24, 1, paths[1])
    if target == 'onet':
        return Detector(O_Net, 48, 1, paths[2])

def GetPrediction(model, target, datatype, size):
    assert target in ['pnet', 'rnet', 'onet']
    assert datatype in ['pos', 'neg']
    filename = '%s_%s.tfrecord' %(target, datatype)
    filename = join(cfg.path_output_files, filename)
    record_iterator = tf.python_io.tf_record_iterator(filename)  
    count = 0
    labels = []
    rankings = []
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        label = int(example.features.feature['image/label'].int64_list.value[0])
        img_string = (example.features.feature['image/encoded'].bytes_list.value[0])
        img_1d = np.fromstring(img_string, dtype=np.uint8)
        if target == 'pnet':
            img = img_1d.reshape(cfg.resize[target], cfg.resize[target], 3)
        else:
            img = img_1d.reshape(1, cfg.resize[target], cfg.resize[target], 3)
        img = (img - 127.5) / 128
        results = model.predict(img)
        cls = results[0]
        labels.append(label)
        rankings.append(cls.reshape(-1)[-1])
        count += 1
        if count >= size: break
    # rankings = np.array(rankings)
    # labels = np.array(labels)
    return rankings, labels

def CalAcc(labels, pred):
    acc = np.sum(labels==pred) / len(pred)
    return acc
    
def CalRecall(labels, pred):
    useful = labels==1
    positive_class = pred[useful]
    return np.sum(positive_class) / len(positive_class)

def Evaluate(labels, prediction, threshold):
    pred = prediction > threshold
    pred = pred.astype(np.int)
    acc = CalAcc(labels, pred)
    recall = CalRecall(labels, pred)
    print('acc is {}'.format(acc))
    print('recall is {}'.format(recall))
    return acc, recall
  

def GetGoodRecall(labels, score):
    def Cost(fpr, tpr):
        # since recall is significant, triple true positive cost
        # return (1-tpr) + fpr
        return (1-tpr) * 3 + fpr
    fpr, tpr, thresholds = metrics.roc_curve(labels, score)
    cost = np.inf
    threshold = None
    for idx in range(len(thresholds)):
        current_cost = Cost(fpr[idx], tpr[idx])
        if current_cost < cost:
            cost = current_cost
            threshold = thresholds[idx]
    return threshold
    
def GetBestAcc(labels, score):
    _, _, thresholds = metrics.roc_curve(labels, score)
    best_acc = 0
    best_threshold = None
    for threshold in thresholds:
        pred = score > threshold
        acc = np.sum(pred==labels) / len(pred)
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
    return best_threshold
    
def GetBestThreshold(labels, score, mode='prediction'):
    assert mode in ['prediction', 'data_gen']
    if mode == 'prediction':
        return GetBestAcc(labels, score)
    else:
        return GetGoodRecall(labels, score)
    
def ShowROC(fpr, tpr):
    import matplotlib.pyplot as plt
    plt.plot(fpr, tpr)
    plt.title('ROC')
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.legend(loc="best")
    plt.show()
    
target = 'pnet'
model = GetModel(target)
size = 1000
prob_pos, labels_pos = GetPrediction(model, target, 'pos', size)
prob_neg, labels_neg = GetPrediction(model, target, 'neg', size*3)
prob = np.array(prob_pos + prob_neg)
labels = np.array(labels_pos + labels_neg)

threshold_prediction = GetBestThreshold(labels, prob)
print('best threshold for prediction is {}'.format(threshold_prediction))
Evaluate(labels, prob, threshold=threshold_prediction)

threshold_data_gen = GetBestThreshold(labels, prob, mode='data_gen')
print('best threshold for data generation is {}'.format(threshold_data_gen))
Evaluate(labels, prob, threshold=threshold_data_gen)

'''
params for vehicle detection
netname     task            acc     recall  threshold
pnet        prediction      0.95    0.92    0.45
pnet        example_gen     0.91    0.98    0.09
rnet        prediction      0.95    0.88    0.47
rnet        example_gen     0.92    0.96    0.13
onet        prediction      0.976   0.95    0.38
'''

'''
params for face detection
netname     task            acc     recall  threshold
pnet        prediction      0.91    0.65    0.9
pnet        example_gen     0.93    0.84    0.4
rnet        prediction      0.97    0.90    0.6
rnet        example_gen     0.90    0.97    0.05
onet        prediction      0.98    0.93    0.7
'''
