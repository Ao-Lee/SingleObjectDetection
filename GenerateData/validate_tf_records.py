import tensorflow as tf
import sys
import numpy as np
from os.path import join
sys.path.append('..')
from config import cfg


def CheckFile(filename, expected_size):
    path = join(cfg.path_output_files, filename)
    record_iterator = tf.python_io.tf_record_iterator(path=path)
    total = 0
    print('begin to check file {}'.format(filename))
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        label = int(example.features.feature['image/label'].int64_list.value[0])
        img_string = (example.features.feature['image/encoded'].bytes_list.value[0])
        img_1d = np.fromstring(img_string, dtype=np.uint8)
        total += 1
        assert label in [0, 1, -1, -2]
        assert len(img_1d) == expected_size**2*3
        if total % 20000 == 0:
            print('number of images checked: {}'.format(total))

    print('there are total {} images in {}'.format(total, filename))
    print('no bugs found in {}'.format(filename))

def Run(target):
    assert target in ['pnet', 'rnet', 'onet']
    types = ['part', 'neg', 'pos', 'landmark']
    for t in types:
        filename = '%s_%s.tfrecord' %(target, t)
        CheckFile(filename, expected_size=cfg.resize[target])
        
def ValidateTFrecords_Pnet():
    Run('pnet')

def ValidateTFrecords_Rnet():
    Run('rnet')

def ValidateTFrecords_Onet():
    Run('onet')

if __name__=='__main__':
    ValidateTFrecords_Onet()


