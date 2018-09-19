import cv2
import numpy as np
import sys
import math

sys.path.append('..')
from config import cfg

'''
Pnet:
	# im_resized        shape(W, H, 3)
	# prob              shape(W', H', 2)
	# reg               shape(W', H', 4)
	prob, reg = self.pnet_detector.predict(im_resized)
Rnet:
	# cropped_ims       shape(K, 24, 24, 3)
	# prob              shape(K, 2)
	# reg               shape(K, 4)
	prob, reg, _ = self.rnet_detector.predict(cropped_ims)
Onet:
	# cropped_ims       shape(K, 48, 48, 3)
	# prob              shape(K, 2)
	# reg               shape(K, 4)
	# landmark          shape(K, 10)
	prob, reg, landmark = self.onet_detector.predict(cropped_ims)
'''

def convert_to_square(bbox):
    square_bbox = bbox.copy()
    h = bbox[:, 3] - bbox[:, 1] + 1
    w = bbox[:, 2] - bbox[:, 0] + 1
    max_side = np.maximum(h, w)
    square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
    square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
    square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
    return square_bbox

# bbox: numpy array, shape n x 5
# reg:  numpy array, shape n x 4
def calibrate_box(bbox, reg):
    w = bbox[:, 2] - bbox[:, 0] + 1
    w = np.expand_dims(w, 1)
    h = bbox[:, 3] - bbox[:, 1] + 1
    h = np.expand_dims(h, 1)
    reg_m = np.hstack([w, h, w, h])
    aug = reg_m * reg
    bbox[:, 0:4] = bbox[:, 0:4] + aug
    return bbox
    
def pad(bboxes, w, h):
    """
        pad the the bboxes, alse restrict the size of it
    Parameters:
    ----------
        bboxes: numpy array, n x 5
            input bboxes
        w: float number
            width of the input image
        h: float number
            height of the input image
    Returns :
    ------
        dy, dx : numpy array, n x 1
            start point of the bbox in target image
        edy, edx : numpy array, n x 1
            end point of the bbox in target image
        y, x : numpy array, n x 1
            start point of the bbox in original image
        ex, ex : numpy array, n x 1
            end point of the bbox in original image
        tmph, tmpw: numpy array, n x 1
            height and width of the bbox
    """
    tmpw, tmph = bboxes[:, 2] - bboxes[:, 0] + 1, bboxes[:, 3] - bboxes[:, 1] + 1
    num_box = bboxes.shape[0]

    dx, dy = np.zeros((num_box,)), np.zeros((num_box,))
    edx, edy = tmpw.copy() - 1, tmph.copy() - 1

    x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

    tmp_index = np.where(ex > w - 1)
    edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
    ex[tmp_index] = w - 1

    tmp_index = np.where(ey > h - 1)
    edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
    ey[tmp_index] = h - 1

    tmp_index = np.where(x < 0)
    dx[tmp_index] = 0 - x[tmp_index]
    x[tmp_index] = 0

    tmp_index = np.where(y < 0)
    dy[tmp_index] = 0 - y[tmp_index]
    y[tmp_index] = 0

    return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
    return_list = [item.astype(np.int32) for item in return_list]

    return return_list
    
# boxes:    shape (K, 5)
def nms(boxes, overlap_threshold, mode='Union'):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(score)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        inter = w * h
        if mode == 'Minimum':
            overlap = inter / np.minimum(area[i], area[idxs[:last]])
        else:
            overlap = inter / (area[i] + area[idxs[:last]] - inter)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))
    return pick
    
################################################# 
def generate_bbox(prob, reg, scale, threshold):
    # prob             (h, w)
    # reg              (h, w, 4)
    stride = 2
    cellsize = 12
    t_index = np.where(prob > threshold)
    if t_index[0].size == 0: return np.array([])

    dx1, dy1, dx2, dy2 = [reg[t_index[0], t_index[1], i] for i in range(4)]

    reg = np.array([dx1, dy1, dx2, dy2])            #(4, K)
    score = prob[t_index[0], t_index[1]]            #(K,)
    boundingbox = np.vstack([np.round((stride * t_index[1]) / scale),
                             np.round((stride * t_index[0]) / scale),
                             np.round((stride * t_index[1] + cellsize) / scale),
                             np.round((stride * t_index[0] + cellsize) / scale),
                             score,
                             reg])

    return boundingbox.T
    
    
def preprocess_inputs(img, scale):
    height, width, channels = img.shape
    new_height = int(height * scale)  # resized new height
    new_width = int(width * scale)  # resized new width
    new_dim = (new_width, new_height)
    img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)  # resized image
    img_resized = (img_resized - 127.5) / 128
    return img_resized

    
# Returns: bboxes shape (K, 9)
def detect_first_stage(img, net, scale, threshold):

    height, width, _ = img.shape
    hs = int(math.ceil(height * scale))
    ws = int(math.ceil(width * scale))
    
    im_data = cv2.resize(img, (ws,hs))
    
    # adjust for the network input
    input_buf = preprocess_inputs(im_data)
    
    
    # input_buf         shape(1, 3, W, H)
    # reg               shape(1, 4, W', H')
    # prob              shape(1, 2, W', H')
    output = net.predict(input_buf)
    reg, prob = output = output
    
    reg = generate_bbox(prob[0,1,:,:], reg, scale, threshold)
    if reg.size == 0: return None

    # nms
    pick = nms(reg[:,0:5], 0.5, mode='Union')
    reg = reg[pick]
    return reg
    
class MtcnnDetector(object):
    def __init__(self,
                 detectors,
                 min_face_size=25,
                 threshold=[0.6, 0.7, 0.7],
                 scale_factor=0.709,
                 nms_pnet = 0.7,
                 nms_rnet = 0.6,
                 nms_onet = 0.6,
                 ):
        self.pnet_detector = detectors[0]
        self.rnet_detector = detectors[1]
        self.onet_detector = detectors[2]
        self.minsize = float(min_face_size)
        self.factor = float(scale_factor)
        self.threshold = threshold
        self.nms_pnet = nms_pnet
        self.nms_rnet = nms_rnet
        self.nms_onet = nms_onet
        
    '''
    inputs:
        img:            shape (h, w, 3)
    outputs:
        total_boxes:    shape (K, 5)
        points:         shape (K, 10)
    '''
    def detect(self, img):
        if img is None: return None
        if len(img.shape) != 3: return None

        MIN_DET_SIZE = cfg.resize['rnet']
        height, width, _ = img.shape
        minl = min(height, width)

        # get all the valid scales
        scales = []
        m = MIN_DET_SIZE/self.minsize
        minl *= m
        factor_count = 0
        while minl > MIN_DET_SIZE:
            scales.append(m*self.factor**factor_count)
            minl *= self.factor
            factor_count += 1

        #############################################
        # first stage
        #############################################
        
        # detected boxes
        total_boxes = []
        for scale in scales:
            local_boxes = detect_first_stage(img, self.Pnet, scale, self.threshold[0])
            if local_boxes is None: continue
            total_boxes.extend(local_boxes)

        if len(total_boxes) == 0: return None
        
        total_boxes = np.vstack(total_boxes)     # (K, 9)

        if total_boxes.size == 0: return None

        # merge the detection from first stage
        pick = nms(total_boxes[:, 0:5], 0.7, 'Union')
        total_boxes = total_boxes[pick]

        bbw = total_boxes[:, 2] - total_boxes[:, 0] + 1
        bbh = total_boxes[:, 3] - total_boxes[:, 1] + 1

        # refine the bboxes
        total_boxes = np.vstack([total_boxes[:, 0]+total_boxes[:, 5] * bbw,
                                 total_boxes[:, 1]+total_boxes[:, 6] * bbh,
                                 total_boxes[:, 2]+total_boxes[:, 7] * bbw,
                                 total_boxes[:, 3]+total_boxes[:, 8] * bbh,
                                 total_boxes[:, 4]
                                 ])

        total_boxes = total_boxes.T
        total_boxes = convert_to_square(total_boxes)
        total_boxes[:, 0:4] = np.round(total_boxes[:, 0:4])
        
        '''
        #############################################
        # second stage
        #############################################
        num_box = total_boxes.shape[0]

        # pad the bbox
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(total_boxes, width, height)
        # (3, 24, 24) is the input shape for RNet
        input_buf = np.zeros((num_box, 3, 24, 24), dtype=np.float32)

        for i in range(num_box):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i]+1, dx[i]:edx[i]+1, :] = img[y[i]:ey[i]+1, x[i]:ex[i]+1, :]
            input_buf[i, :, :, :] = preprocess_inputs(cv2.resize(tmp, (24, 24)))

            
        # input_buf         shape(K, 3, 24, 24)
        # reg               shape(K, 4)
        # prob              shape(K, 2)
        output = self.RNet.predict(input_buf)
        reg, prob = output
        
        # filter the total_boxes with threshold
        passed = np.where(prob[:, 1] > self.threshold[1])
        total_boxes = total_boxes[passed]

        if total_boxes.size == 0: return None

        total_boxes[:, 4] = prob[passed, 1].reshape((-1,))
        reg = reg[passed]

        # nms
        pick = nms(total_boxes, 0.7, 'Union')
        total_boxes = total_boxes[pick]
        reg = reg[pick]

        total_boxes = calibrate_box(total_boxes, reg)
        total_boxes = convert_to_square(total_boxes)
        total_boxes[:, 0:4] = np.round(total_boxes[:, 0:4])

        #############################################
        # third stage
        #############################################
        num_box = total_boxes.shape[0]

        # pad the bbox
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(total_boxes, width, height)
        # (3, 48, 48) is the input shape for ONet
        input_buf = np.zeros((num_box, 3, 48, 48), dtype=np.float32)

        for i in range(num_box):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.float32)
            tmp[dy[i]:edy[i]+1, dx[i]:edx[i]+1, :] = img[y[i]:ey[i]+1, x[i]:ex[i]+1, :]
            input_buf[i, :, :, :] = preprocess_inputs(cv2.resize(tmp, (48, 48)))

        # input_buf         shape(K, 3, 48, 48)
        # points            shape(K, 10)
        # reg               shape(K, 4)
        # prob              shape(K, 2)
        output = self.ONet.predict(input_buf)
        points, reg, prob = output

        # filter the total_boxes with threshold
        passed = np.where(prob[:, 1] > self.threshold[2])
        total_boxes = total_boxes[passed]

        if total_boxes.size == 0: return None

        total_boxes[:, 4] = prob[passed, 1].reshape((-1,))
        reg = reg[passed]
        points = points[passed]

        # compute landmark points
        bbw = total_boxes[:, 2] - total_boxes[:, 0] + 1
        bbh = total_boxes[:, 3] - total_boxes[:, 1] + 1
        points[:, 0:5] = np.expand_dims(total_boxes[:, 0], 1) + np.expand_dims(bbw, 1) * points[:, 0:5]
        points[:, 5:10] = np.expand_dims(total_boxes[:, 1], 1) + np.expand_dims(bbh, 1) * points[:, 5:10]

        # nms
        total_boxes = calibrate_box(total_boxes, reg)
        pick = nms(total_boxes, 0.7, 'Min')
        total_boxes = total_boxes[pick]
        points = points[pick]
        
        return total_boxes, points
        '''
   
'''      
class MtcnnDetector(object):
    def __init__(self,
                 detectors,
                 min_face_size=25,
                 stride=2,
                 threshold=[0.6, 0.7, 0.7],
                 scale_factor=0.79,
                 nms_pnet = 0.7,
                 nms_rnet = 0.6,
                 nms_onet = 0.6,
                 ):

        self.pnet_detector = detectors[0]
        self.rnet_detector = detectors[1]
        self.onet_detector = detectors[2]
        self.min_face_size = min_face_size
        self.stride = stride
        self.thresh = threshold
        self.scale_factor = scale_factor
        self.nms_pnet = nms_pnet
        self.nms_rnet = nms_rnet
        self.nms_onet = nms_onet

 


    # im shape (H, W, 3)
    # total_boxes shape (K, 5)
    def detect_pnet(self, im):
        h, w, c = im.shape
        net_size = 12
        current_scale = float(net_size) / self.min_face_size  
        im_resized = preprocess_inputs(im, current_scale)
        current_height, current_width, _ = im_resized.shape
        total_boxes = []

        while min(current_height, current_width) > net_size:

            # im_resized        shape(W, H, 3)
            # prob              shape(W', H', 2)
            # reg               shape(W', H', 4)
            prob, reg = self.pnet_detector.predict(im_resized)
          
            #boxes: num*9(x1,y1,x2,y2,score,x1_offset,y1_offset,x2_offset,y2_offset)
            boxes = generate_bbox(prob[:, :,1], reg, current_scale, self.thresh[0])
            current_scale *= self.scale_factor
            im_resized = preprocess_inputs(im, current_scale)
            current_height, current_width, _ = im_resized.shape

            if boxes.size == 0: continue
            keep = nms(boxes[:, :5], 0.5, 'Union')
            boxes = boxes[keep]
            total_boxes.append(boxes)
            

        if len(total_boxes) == 0: return None
        
        total_boxes = np.vstack(total_boxes)        # shape (K, 9)

        keep = nms(total_boxes[:, 0:5], self.nms_pnet, 'Union')
        total_boxes = total_boxes[keep]
        boxes = total_boxes[:, :5]

        bbw = total_boxes[:, 2] - total_boxes[:, 0] + 1
        bbh = total_boxes[:, 3] - total_boxes[:, 1] + 1

        total_boxes = np.vstack([total_boxes[:, 0] + total_boxes[:, 5] * bbw,
                             total_boxes[:, 1] + total_boxes[:, 6] * bbh,
                             total_boxes[:, 2] + total_boxes[:, 7] * bbw,
                             total_boxes[:, 3] + total_boxes[:, 8] * bbh,
                             total_boxes[:, 4]])

        return total_boxes.T
        
    # im shape (H, W, 3)
    # dets shape (K, 5)
    # boxes: shape (M, 5)
    def detect_rnet(self, im, dets):
        h, w, c = im.shape
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
        num_boxes = dets.shape[0]
        cropped_ims = np.zeros((num_boxes, 24, 24, 3), dtype=np.float32)
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (24, 24))-127.5) / 128


        # cropped_ims       shape(K, 24, 24, 3)
        # prob              shape(K, 2)
        # reg               shape(K, 4)
        prob, reg, _ = self.rnet_detector.predict(cropped_ims)
        prob = prob[:,1]
        passed = np.where(prob > self.thresh[1])[0]

        if len(passed) == 0: return None

        boxes = dets[passed]
        boxes[:, 4] = prob[passed]
        reg = reg[passed]
   

        pick = nms(boxes, self.nms_rnet)
        boxes = boxes[pick]
        boxes = calibrate_box(boxes, reg[pick])
        return boxes
    
    # im:           shape (H, W, 3)
    # dets:         shape (K, 5)
    # boxes         shape (M, 5)
    # landmark:     shape (M, 10)
    def detect_onet(self, im, dets):
        h, w, c = im.shape
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
        num_boxes = dets.shape[0]
        cropped_ims = np.zeros((num_boxes, 48, 48, 3), dtype=np.float32)
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (48, 48))-127.5) / 128
            
        
        # cropped_ims       shape(K, 48, 48, 3)
        # prob              shape(K, 2)
        # reg               shape(K, 4)
        # landmark          shape(K, 10)
        prob, reg, landmark = self.onet_detector.predict(cropped_ims)
        prob = prob[:,1]        
        passed = np.where(prob > self.thresh[2])[0] 

        if len(passed) == 0 : return None, None       

        boxes = dets[passed]
        boxes[:, 4] = prob[passed]
        reg = reg[passed]
        landmark = landmark[passed]
       
        
        #width
        w = boxes[:,2] - boxes[:,0] + 1
        #height
        h = boxes[:,3] - boxes[:,1] + 1
        landmark[:,0::2] = (np.tile(w,(5,1)) * landmark[:,0::2].T + np.tile(boxes[:,0],(5,1)) - 1).T
        landmark[:,1::2] = (np.tile(h,(5,1)) * landmark[:,1::2].T + np.tile(boxes[:,1],(5,1)) - 1).T        
        

        boxes = calibrate_box(boxes, reg)
        pick = nms(boxes, self.nms_onet, 'Minimum')
        boxes = boxes[pick]
        landmark = landmark[pick]
        return boxes, landmark
    

    def detect(self, img):

        if self.pnet_detector:
            boxes = self.detect_pnet(img)
            if boxes is None: return np.array([]),np.array([])

        if self.rnet_detector:
            boxes = self.detect_rnet(img, boxes)
            if boxes is None: return np.array([]),np.array([])
    
        if self.onet_detector:
            boxes,landmark = self.detect_onet(img, boxes)
            if boxes is None: return np.array([]),np.array([])  
        return boxes,landmark
'''        
        
        
'''
# boxes:    shape (K, 5)
def fast_nms(boxes, overlap_threshold, mode="Union"):
    print(boxes.shape)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == "Union":
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == "Minimum":
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        #keep
        inds = np.where(ovr <= overlap_threshold)[0]
        order = order[inds + 1]

    return keep
'''
        