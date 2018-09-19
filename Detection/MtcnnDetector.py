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

nms_mode = {'Union':1, 'Minimum':0}

# boxes:    shape (K, 5)
def nms(boxes, overlap_threshold, mode=nms_mode['Union']):
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
        if mode == nms_mode['Minimum']:
            overlap = inter / np.minimum(area[i], area[idxs[:last]])
        else:
            overlap = inter / (area[i] + area[idxs[:last]] - inter)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))
    return pick

# in_data:      shape(h, w, c)
# out_data:     shape(h, w, c)
def preprocess_inputs(in_data):

    if in_data.dtype is not np.dtype('float32'):
        out_data = in_data.astype(np.float32)
    else:
        out_data = in_data

    out_data = (out_data - 127.5) / 128
    return out_data
        
class MtcnnDetector(object):
    def __init__(self,
                 detectors,
                 minsize = 20,
                 factor = 0.709,
                 thresh_prediction = [0.6, 0.7, 0.8],
                 thresh_nms = [0.5, 0.7, 0.7],
                 modes = [nms_mode['Union'], nms_mode['Union'], nms_mode['Minimum']],
                 thresh_merge = 0.7,
                 mode_merge = nms_mode['Union']
                 ):
        
        self.Pnet = detectors[0]
        self.Rnet = detectors[1]
        self.Onet = detectors[2]
        self.minsize = float(minsize)
        self.factor = float(factor)
        self.thresh_prediction = thresh_prediction
        self.thresh_nms = thresh_nms
        self.thresh_merge = thresh_merge
        self.modes = modes
        self.mode_merge = mode_merge
        
    '''
    inputs:
        img:            shape (h, w, 3)
    outputs:
        total_boxes:    shape (K, 5)
        points:         shape (K, 10)
    '''
    def detect(self, img):
        if img is None: return self.EmptyResult()
        if len(img.shape) != 3: return self.EmptyResult()

        height, width, _ = img.shape
        minl = min(height, width)

        # get all the valid scales
        scales = []
        m = cfg.resize['rnet']/self.minsize
        minl *= m
        factor_count = 0
        while minl > cfg.resize['rnet']:
            scales.append(m*self.factor**factor_count)
            minl *= self.factor
            factor_count += 1

        #############################################
        # first stage
        #############################################
        
        # detected boxes
        total_boxes = []
        for scale in scales:
            # def detect_first_stage(self, img, net, scale, threshold, thresh_nms=0.5, mode_nms='Union'):
            local_boxes = self.detect_first_stage(img, self.Pnet, scale, self.thresh_prediction[0], self.thresh_nms[0], self.modes[0])
            if local_boxes is None: continue
            total_boxes.extend(local_boxes)

        if len(total_boxes) == 0: return self.EmptyResult()
        
        total_boxes = np.vstack(total_boxes)     # (K, 9)

        if total_boxes.size == 0: return self.EmptyResult()

        # merge the detection from first stage
        pick = nms(total_boxes[:, 0:5], self.thresh_merge, self.mode_merge)
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
        total_boxes = self.convert_to_square(total_boxes)
        total_boxes[:, 0:4] = np.round(total_boxes[:, 0:4])     # (K, 5)
        
        
        #############################################
        # second stage
        #############################################
        num_box = total_boxes.shape[0]

        # pad the bbox
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(total_boxes, width, height)
        # this is the input shape for Rnet
        input_buf = np.zeros((num_box, 24, 24, 3), dtype=np.float32)

        for i in range(num_box):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i]+1, dx[i]:edx[i]+1, :] = img[y[i]:ey[i]+1, x[i]:ex[i]+1, :]
            input_buf[i, :, :, :] = preprocess_inputs(cv2.resize(tmp, (24, 24)))


         
        # input_buf         shape(K, 24, 24, 3)
        # reg               shape(K, 4)
        # prob              shape(K, 2)
        prob, reg, _ = self.Rnet.predict(input_buf)

        
        # filter the total_boxes with threshold
        passed = np.where(prob[:, 1] > self.thresh_prediction[1])
        total_boxes = total_boxes[passed]

        if total_boxes.size == 0: return self.EmptyResult()

        total_boxes[:, 4] = prob[passed, 1].reshape((-1,))
        reg = reg[passed]

        # nms
        pick = nms(total_boxes, self.thresh_nms[1], self.modes[1])
        total_boxes = total_boxes[pick]
        reg = reg[pick]

        total_boxes = self.calibrate_box(total_boxes, reg)
        total_boxes = self.convert_to_square(total_boxes)
        total_boxes[:, 0:4] = np.round(total_boxes[:, 0:4])
        
        
        #############################################
        # third stage
        #############################################
        num_box = total_boxes.shape[0]

        # pad the bbox
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(total_boxes, width, height)
        # this is the input shape for Onet
        input_buf = np.zeros((num_box, 48, 48, 3), dtype=np.float32)

        for i in range(num_box):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.float32)
            tmp[dy[i]:edy[i]+1, dx[i]:edx[i]+1, :] = img[y[i]:ey[i]+1, x[i]:ex[i]+1, :]
            input_buf[i, :, :, :] = preprocess_inputs(cv2.resize(tmp, (48, 48)))

        # input_buf         shape(K, 3, 48, 48)
        # points            shape(K, 10)
        # reg               shape(K, 4)
        # prob              shape(K, 2)
        prob, reg, points = self.Onet.predict(input_buf)

        # filter the total_boxes with threshold
        passed = np.where(prob[:, 1] > self.thresh_prediction[2])
        total_boxes = total_boxes[passed]

        if total_boxes.size == 0: return self.EmptyResult()

        total_boxes[:, 4] = prob[passed, 1].reshape((-1,))
        reg = reg[passed]
        points = points[passed]

        # compute landmark points
        bbw = total_boxes[:, 2] - total_boxes[:, 0] + 1
        bbh = total_boxes[:, 3] - total_boxes[:, 1] + 1
        points[:, 0:5] = np.expand_dims(total_boxes[:, 0], 1) + np.expand_dims(bbw, 1) * points[:, 0:5]
        points[:, 5:10] = np.expand_dims(total_boxes[:, 1], 1) + np.expand_dims(bbh, 1) * points[:, 5:10]

        # nms
        total_boxes = self.calibrate_box(total_boxes, reg)
        pick = nms(total_boxes, self.thresh_nms[2], self.modes[2])
        total_boxes = total_boxes[pick]
        points = points[pick]
        
        return total_boxes, points
        
    def EmptyResult(self):
        total_boxes = np.zeros(shape=[0,5])
        points = np.zeros(shape=[0,10])
        return total_boxes, points
        
    def convert_to_square(self, bbox):
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
    def calibrate_box(self, bbox, reg):
        w = bbox[:, 2] - bbox[:, 0] + 1
        w = np.expand_dims(w, 1)
        h = bbox[:, 3] - bbox[:, 1] + 1
        h = np.expand_dims(h, 1)
        reg_m = np.hstack([w, h, w, h])
        aug = reg_m * reg
        bbox[:, 0:4] = bbox[:, 0:4] + aug
        return bbox
        
    def pad(self, bboxes, w, h):
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
        

    # prob:     (h, w)
    # reg:      (h, w, 4)
    # return:   (K, 9) of form (x1,y1,x2,y2,score,x1_offset,y1_offset,x2_offset,y2_offset) 
    def generate_bbox(self, prob, reg, scale, threshold):
    
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
        
    # Returns: bboxes shape (K, 9)
    def detect_first_stage(self, img, net, scale, threshold, thresh_nms=0.5, mode_nms='Union'):
    
        height, width, _ = img.shape
        hs = int(math.ceil(height * scale))
        ws = int(math.ceil(width * scale))
        
        im_data = cv2.resize(img, (ws,hs))
        
        # adjust for the network input
        input_buf = preprocess_inputs(im_data)
        
        
        # input_buf         shape(H, W, C)
        # prob              shape(H', W', 2)
        # reg               shape(H', W', 4)
        prob, reg = net.predict(input_buf)
        
        reg = self.generate_bbox(prob[:,:,1], reg, scale, threshold)
        if reg.size == 0: return None
    
        # nms
        pick = nms(reg[:,0:5], 0.5, mode='Union')
        reg = reg[pick]
        return reg
   
      