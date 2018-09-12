import numpy as np

'''
Compute average precision given each result of prediction and the confidence

Parameters:
    prediction:     a list of dictionary. 
                    for example [{'TP':True, 'confidence':0.65}, {'TP':False, 'confidence':0.4}]
                    set TP to True if the prediction is true positive
                    set TP to False if the prediction is false positive
                    
    num_gt_example: int, number of ground truth examples
    
    validate:       bool, turn it on to check the inputs

Returns:
    ap:             int, average precision. note that we assume the inputs belongs to the same class
                    it is possible to utilize this functionality to compute mean average precision via multiple classes
                    
    recall:         recalls computed over different confidence threshold, numpy array of shape (N,)
    
    precision:      precision computed over different confidence threshold, numpy array of shape (N,) 
    
'''
def VocAveragePrecision(prediction, num_gt_example, validate=False):
    def CheckInput(prediction, num_gt_example):
        num_true_positive = 0
        for d in prediction:
            keys = list(d.keys())
            assert 'TP' in keys and 'confidence' in keys and type(d['TP']) is bool
            assert len(keys) == 2
            if d['TP']: num_true_positive +=1
        assert num_true_positive <= num_gt_example
        return
    
    if validate: CheckInput(prediction, num_gt_example)
    
    prediction.sort(key=lambda x:x['confidence'], reverse=True)
    prediction = [info['TP'] for info in prediction]
    prediction = np.array(prediction)
    
    tp = (prediction==True).astype(np.int)
    fp = (prediction==False).astype(np.int)
    
    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)
    
    # precision_scores = cumulative_tp / (np.arange(len(prediction)) + 1)
    precision = cumulative_tp / (cumulative_tp + cumulative_fp)
    recall = cumulative_tp / num_gt_example
    
    '''
    metric      rank(confidence)    extream_value
    precision   inf                 1
    precision   -inf                0
    recall      inf                 0
    recall      -inf                1
    '''
    precision = np.array([1.0] + list(precision) + [0.0])
    recall = np.array([0.0] + list(recall) + [1.0])
    
    for i in range(len(precision)-2, -1, -1):
        precision[i] = max(precision[i], precision[i+1])

    i_list = []
    for i in range(1, len(recall)):
        if recall[i] != recall[i-1]:
            i_list.append(i)
    
    ap = 0.0
    for i in i_list:
        ap += ((recall[i]-recall[i-1])*precision[i])
    return ap, recall, precision
   
'''
Compute IoU between detect box and ground truth boxes

Parameters:
    box:            detection box, numpy array, shape (4,): x1, y1, x2, y2
    
    boxes:          ground truth boxes, shape (n, 4): x1, y1, x2, y2

Returns:
    iou:            numpy array, shape (n, )
'''
def IoU(box, boxes):
    
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    
    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    iou = inter / (box_area + area - inter)
    return iou


'''
Compute average precision given predictions of images

Parameters:     
    gt_info:        a dictionary of files, representing the ground truth infomation 
                    of each image, for example
                        {'1.jpg':bboxes_01, '2.jpg':bboxes_02}
                    each key represents each image file, and the bboxes_01 is 
                    the numpy array of shape (K, 4), representing K ground truth
                    objects.
                    
    predictions_info:
                    a list, prepresenting each prediction, for example: [prediction01, prediction02]
                    each prediction is a dictionary, for example
                        {'bbox': array([165., 214., 230., 255.]), 'confidence':0.85, 'file_id':'2.jpg'}
                    bbox is numpy array of shape (4, )
                    file_id should be consistent with the files in gt_info
                    
    num_gt_example: int, number of ground truth examples
    
    validate_input: bool, turn it on to check the inputs
    
    min_overlap:    should be in (0, 1) default value is 0.5 (defined in the PASCAL VOC2012 challenge)
                    
Returns:
    ap:             int, average precision. note that we assume the inputs belongs to the same class
                    it is possible to utilize this functionality to compute mean average precision via multiple classes
                    
    recall:         recalls computed over different confidence threshold, numpy array of shape (N,)
    
    precision:      precision computed over different confidence threshold, numpy array of shape (N,) 
    
'''
def AveragePrecisionOnImages(gt_info, predictions_info, num_gt_example, min_overlap=0.5, validate_input=False):
    result = []
    used_info = {}
    # set 'used' tag for each ground truch bbox
    for file_id in gt_info:
        boxes = gt_info[file_id]
        size = boxes.shape[0]
        used = np.zeros(shape=[size]).astype(np.bool)
        used_info[file_id] = used

    for idx, prediction in enumerate(predictions_info):
        file_id = prediction['file_id']
        box_pred = prediction['bbox']

        boxes_gt = gt_info[file_id]
        iou = IoU(box_pred, boxes_gt)
        idx = iou.argmax()
        used = used_info[file_id][idx]
        confidence = prediction['confidence']

        if iou[idx] >= min_overlap and not used:
            result.append({'TP':True, 'confidence':confidence})
            used_info[file_id][idx] = True
        else:
            result.append({'TP':False, 'confidence':confidence})

    ap, recall, precision = VocAveragePrecision(result, num_gt_example, validate=validate_input)
    return ap, recall, precision
    
    
def _GenRandomBoxes(size=10):
    def _GenRandomBox():
        rand = np.random.randint(low=0, high=101, size=4)
        box = np.array([rand[0], rand[1], rand[0]+rand[2], rand[1]+rand[3]])
        box[box<0]=0
        return box

    arrays = []
    for _ in range(size):
        arrays.append(_GenRandomBox())
    return np.stack(arrays)
    
def _GenPredictionBox(box, fp=False):
    w = box[2] - box[0]
    h = box[3] - box[1]
    dw = int(w*0.7) if fp else int(w*0.3)
    dh = int(h*0.7) if fp else int(h*0.3)
    rand_x = np.random.randint(low=-dw, high=dw+1, size=2)
    rand_y = np.random.randint(low=-dh, high=dh+1, size=2)
    new_box = [box[0]+rand_x[0], box[1]+rand_y[0], box[2]+rand_x[1], box[3]+rand_y[1]]

    new_box = np.array(new_box)
    new_box[new_box<0]=0
    if new_box[2]<new_box[0]: new_box[2]=new_box[0]+1
    if new_box[3]<new_box[1]: new_box[3]=new_box[1]+1

    return new_box

def _GenFileNames(size=100):
    result = []
    while len(result) < size:
        n = np.random.randint(low=0, high=10000)
        s = str(n).zfill(4)
        s+='.jpg'
        if s not in result:
            result.append(s)
    return result
    
def _GenImgData(file_id):
    num_gt = np.random.randint(low=1, high=6)
    gt_boxes = _GenRandomBoxes(size=num_gt)
    gt_key = file_id
    gt_value = gt_boxes

    prediction = []
    for idx in range(num_gt):
        if np.random.random() < 0.2:
            continue
        fp = True if np.random.random()<0.2 else False
        pred_box = _GenPredictionBox(gt_boxes[idx], fp)
        iou = np.max(IoU(pred_box, gt_boxes))
        dc = (np.random.random()-0.5)/3
        confidence = iou + dc
        result = {'bbox':pred_box, 'confidence':confidence, 'file_id':file_id}
        prediction.append(result)
        
    return gt_key, gt_value, prediction, num_gt
    
def _GenFakeData(size=10):
    files = _GenFileNames(size=size)
    gts = {}
    predictions = []
    num_gt = 0
    for file in files:
        gt_key, gt_value, prediction, num_gt_this_img = _GenImgData(file)
        gts[gt_key] = gt_value
        predictions += prediction
        num_gt += num_gt_this_img
    return gts, predictions, num_gt

if __name__=='__main__':
    gts, predictions, num_gt = _GenFakeData(size=1)
    ap, _, _ = AveragePrecisionOnImages(gts, predictions, num_gt, min_overlap=0.5, validate_input=True)
    print(ap)


    
    
    
    
    