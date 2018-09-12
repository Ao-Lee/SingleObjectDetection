from os.path import join
import random
from tqdm import tqdm
import tensorflow as tf
import sys

try:
    from .tfrecord_utils import _process_image_withoutcoder, _convert_to_example_simple
except ImportError:
    from tfrecord_utils import _process_image_withoutcoder, _convert_to_example_simple
    
sys.path.append('..')
from config import cfg


def _add_to_tfrecord(filename, image_example, tfrecord_writer):
    """Loads data from image and annotations files and add them to a TFRecord.

    Args:
      dataset_dir: Dataset directory;
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    # print('---', filename)
    #imaga_data:array to string
    #height:original image's height
    #width:original image's width
    #image_example dict contains image's info
    filename = join(cfg.path_output_files, filename)
    image_data, height, width = _process_image_withoutcoder(filename)
    example = _convert_to_example_simple(image_example, image_data)
    tfrecord_writer.write(example.SerializeToString())

def get_dataset(target, types):
    path =join(cfg.path_output_txt, '%s_%s.txt'%(target, types))

    with open(path, 'r') as f:
        imagelist = f.readlines()
        
    dataset = []
    for line in imagelist:
        info = line.strip().split(' ')
        data_example = dict()
        bbox = dict()
        data_example['filename'] = info[0]
        data_example['label'] = int(info[1])
        bbox['xmin'] = 0
        bbox['ymin'] = 0
        bbox['xmax'] = 0
        bbox['ymax'] = 0
        bbox['xlefteye'] = 0
        bbox['ylefteye'] = 0
        bbox['xrighteye'] = 0
        bbox['yrighteye'] = 0
        bbox['xnose'] = 0
        bbox['ynose'] = 0
        bbox['xleftmouth'] = 0
        bbox['yleftmouth'] = 0
        bbox['xrightmouth'] = 0
        bbox['yrightmouth'] = 0        
        if len(info) == 6:
            bbox['xmin'] = float(info[2])
            bbox['ymin'] = float(info[3])
            bbox['xmax'] = float(info[4])
            bbox['ymax'] = float(info[5])
        if len(info) == 12:
            bbox['xlefteye'] = float(info[2])
            bbox['ylefteye'] = float(info[3])
            bbox['xrighteye'] = float(info[4])
            bbox['yrighteye'] = float(info[5])
            bbox['xnose'] = float(info[6])
            bbox['ynose'] = float(info[7])
            bbox['xleftmouth'] = float(info[8])
            bbox['yleftmouth'] = float(info[9])
            bbox['xrightmouth'] = float(info[10])
            bbox['yrightmouth'] = float(info[11])
            
        data_example['bbox'] = bbox
        dataset.append(data_example)

    return dataset
    
    
def ConvertToRecords(target, types):
    tf_filename =join(cfg.path_output_files, '%s_%s.tfrecord'%(target, types))
    if tf.gfile.Exists(tf_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return
        
    dataset = get_dataset(target=target, types=types)

    random.shuffle(dataset)
        
    with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
        for image_example in tqdm(dataset):
            filename = image_example['filename']
            _add_to_tfrecord(filename, image_example, tfrecord_writer)
    print('Finished converting {} records!'.format(types))
    
def Run(target):
    assert target in ['pnet', 'rnet', 'onet']
    types = ['pos', 'neg', 'part', 'landmark']
    for current_type in types:
        ConvertToRecords(target, current_type)
        
        
def GenRecordsPnet():
    Run(target='pnet')
    
def GenRecordsRnet():
    Run(target='rnet')
    
def GenRecordsOnet():
    Run(target='onet')
    
if __name__ == '__main__':
    GenRecordsPnet()
    # GenRecordsRnet
    # GenRecordsOnet
        



    

