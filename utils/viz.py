import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.misc import imresize

def ShowImg(img, title=''):
    h = img.shape[0]//150
    w = img.shape[1]//150
    plt.figure(figsize=(w, h))
    plt.imshow(img.astype('uint8'))
    plt.title(title)
    plt.axis('off')
    plt.show()
    plt.close()
    

'''
img: an ndarray of shape (w, h, 3)
bboxes: an ndarray of shape (num, 4)
'''
def PlotImgWithBboxes(img, bboxes, color=(0, 255, 255)):
    for bbox in bboxes:
        left = int(bbox[0])
        top = int(bbox[1])
        right = int(bbox[2])
        bot = int(bbox[3])
        cv2.rectangle(img, (left, top), (right, bot), color, 3)
    return img
    

# if u got a list of imgs with the same size, and u wanna show them together in one shot, here is what u got
def MergeImage(imgs, how='auto', color=(40,40,40), margin='auto', min_size=600):

    assert how in ['vertical', 'horizontal', 'auto']
    num = len(imgs)
    assert num >= 1
    h = imgs[0].shape[0]
    w = imgs[0].shape[1]

    for img in imgs:
        assert img.shape == (h, w, 3)

    if how == 'auto':
        how = 'horizontal' if h < w else 'vertical'
    color = np.array(color,dtype=np.uint8)
    if margin == 'auto':
        margin = min(h, w)//20

    
    new_h = h + margin*2 if how=='horizontal' else h*num + margin*(num+1)
    new_w = w + margin*2 if how=='vertical' else w*num + margin*(num+1)
    
    new_img = np.zeros([new_h, new_w, 3], dtype=np.uint8)
    new_img[:,:,:] = color

    for i, img in enumerate(imgs):
        if how == 'horizontal':
            start = margin*(i+1) + w*i
            end = margin*(i+1) + w*(i+1)
            new_img[margin:margin+h, start:end, :] = img

        if how == 'vertical':
            start = margin*(i+1) + h*i
            end = margin*(i+1) + h*(i+1)
            new_img[start:end, margin:margin+w, :] = img
        
    size = min(new_w, new_h)
    ratio = 1 if size<= min_size else min_size/size
    new_w = int(new_w*ratio)
    new_h = int(new_h*ratio)
    new_img = imresize(new_img, (new_h, new_w))
    return new_img
    
    