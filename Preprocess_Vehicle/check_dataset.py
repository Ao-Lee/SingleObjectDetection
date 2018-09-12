from os.path import join
import sys
import numpy as np
sys.path.append('..')
from config import cfg
from utils.viz import PlotImgWithBboxes, ShowImg
from utils.common import Inputs2ArrayImage

def CheckValidation(annotations):
    for anno in annotations:
        anno = anno.strip().split(sep=' ')
        bboxes = anno[1:]
        alert_info = 'wrong box in image {}'.format(anno[0])
        assert len(bboxes) % 4 == 0, alert_info
        bboxes = np.array(bboxes, dtype=np.int).reshape(-1, 4)
        for bbox in bboxes:
            left, top, right, bot = tuple(bbox)
            alert_info = 'wrong box in image {}'.format(anno[0]) + str([left, top, right, bot])
            assert left < right and top < bot, alert_info 


def MyPlot(annotations):
    indices = np.random.randint(low=0, high=len(annotations), size=5)
    
    for idx in indices:
        content = annotations[idx].strip().split(sep=' ')
        path_img = join(cfg.path_detection_imgs, content[0])
        img = Inputs2ArrayImage(path_img)
        bboxes = content[1:]
        bboxes = np.array(bboxes, dtype=np.int).reshape(-1, 4)
        img = PlotImgWithBboxes(img, bboxes)
        ShowImg(img)

if __name__=='__main__':
    with open(cfg.path_detection_labels, 'r') as f:
        annotations = f.readlines()
    CheckValidation(annotations)   
    MyPlot(annotations)
    