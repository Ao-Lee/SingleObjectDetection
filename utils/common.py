import os
import numpy as np
from PIL import Image


'''
convert an input to PIL image or numpy array image
inputs could be (1)string (2)PIL Image (3) numpy array image
'''
def Inputs2ArrayImage(inputs, output_type=np.ndarray, dtype='uint8', size=None):
    assert output_type in [np.ndarray, Image.Image]

    img = None
    if isinstance(inputs, str): # input is a path
        img = Image.open(os.path.expanduser(inputs))
    elif isinstance(inputs, Image.Image): # input is a PIL image
        pass
    elif isinstance(inputs, np.ndarray): # input is a numpy array
        img = Image.fromarray(np.uint8(inputs))
    else:
        msg = 'unexpected type of input! '
        msg += 'expect str, PIL or ndarray image, '
        msg += 'but got {}'
        raise TypeError(msg.format(type(inputs)))
        
    if size is not None:
        img = img.resize((size, size))
        
    if output_type==np.ndarray:
        img = np.array(img).astype(dtype)
        if len(img.shape)==2:
            img = _Gray2RGB(img)
        
    return img
  
def _Gray2RGB(img):
    assert len(img.shape)==2
    R = np.expand_dims(img, axis=-1)
    G = np.expand_dims(img, axis=-1)
    B = np.expand_dims(img, axis=-1)
    return np.concatenate([R,G,B],axis=-1)
   
