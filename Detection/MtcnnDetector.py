import cv2
import numpy as np
from .nms import py_nms

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

    def convert_to_square(self, bbox):
        """
            convert bbox to square
        Parameters:
        ----------
            bbox: numpy array , shape n x 5
                input bbox
        Returns:
        -------
            square bbox
        """
        square_bbox = bbox.copy()

        h = bbox[:, 3] - bbox[:, 1] + 1
        w = bbox[:, 2] - bbox[:, 0] + 1
        max_side = np.maximum(h, w)
        square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
        square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
        square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
        square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
        return square_bbox

    def calibrate_box(self, bbox, reg):
        """
            calibrate bboxes
        Parameters:
        ----------
            bbox: numpy array, shape n x 5
                input bboxes
            reg:  numpy array, shape n x 4
                bboxes adjustment
        Returns:
        -------
            bboxes after refinement
        """
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
        
    def generate_bbox(self, cls_map, reg, scale, threshold):
        """
            generate bbox from feature cls_map
        Parameters:
        ----------
            cls_map: numpy array , n x m (m=1, n=1 for Pnet) 
                detect score for each position
            reg: numpy array , n x m x 4 (m=1, n=1 for Pnet) 
                bbox
            scale: float number
                scale of this detection
            threshold: float number
                detect threshold
        Returns:
        -------
            bbox array
        """
        stride = 2
        cellsize = 12


        t_index = np.where(cls_map > threshold)

        # find nothing
        if t_index[0].size == 0:
            return np.array([])
        #offset
        dx1, dy1, dx2, dy2 = [reg[t_index[0], t_index[1], i] for i in range(4)]

        reg = np.array([dx1, dy1, dx2, dy2])
        score = cls_map[t_index[0], t_index[1]]
        boundingbox = np.vstack([np.round((stride * t_index[1]) / scale),
                                 np.round((stride * t_index[0]) / scale),
                                 np.round((stride * t_index[1] + cellsize) / scale),
                                 np.round((stride * t_index[0] + cellsize) / scale),
                                 score,
                                 reg])

        return boundingbox.T
    #pre-process images
    def processed_image(self, img, scale):
        height, width, channels = img.shape
        new_height = int(height * scale)  # resized new height
        new_width = int(width * scale)  # resized new width
        new_dim = (new_width, new_height)
        img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)  # resized image
        img_resized = (img_resized - 127.5) / 128
        return img_resized


    # im shape (H, W, 3)
    # total_boxes shape (K, 5)
    def detect_pnet(self, im):
        h, w, c = im.shape
        net_size = 12
        current_scale = float(net_size) / self.min_face_size  
        im_resized = self.processed_image(im, current_scale)
        current_height, current_width, _ = im_resized.shape
        total_boxes = []

        while min(current_height, current_width) > net_size:

            # im_resized        shape(W, H, 3)
            # prob              shape(W', H', 2)
            # reg               shape(W', H', 4)
            prob, reg = self.pnet_detector.predict(im_resized)
          
            #boxes: num*9(x1,y1,x2,y2,score,x1_offset,y1_offset,x2_offset,y2_offset)
            boxes = self.generate_bbox(prob[:, :,1], reg, current_scale, self.thresh[0])
            current_scale *= self.scale_factor
            im_resized = self.processed_image(im, current_scale)
            current_height, current_width, _ = im_resized.shape

            if boxes.size == 0: continue
            keep = py_nms(boxes[:, :5], 0.5, 'Union')
            boxes = boxes[keep]
            total_boxes.append(boxes)
            

        if len(total_boxes) == 0: return None
        
        total_boxes = np.vstack(total_boxes)        # shape (K, 9)

        keep = py_nms(total_boxes[:, 0:5], self.nms_pnet, 'Union')
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
        dets = self.convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
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
   

        pick = py_nms(boxes, self.nms_rnet)
        boxes = boxes[pick]
        boxes = self.calibrate_box(boxes, reg[pick])
        return boxes
    
    # im:           shape (H, W, 3)
    # dets:         shape (K, 5)
    # boxes         shape (M, 5)
    # landmark:     shape (M, 10)
    def detect_onet(self, im, dets):
        h, w, c = im.shape
        dets = self.convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
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
        

        boxes = self.calibrate_box(boxes, reg)
        pick = py_nms(boxes, self.nms_onet, 'Minimum')
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
        