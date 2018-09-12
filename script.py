from GenerateData.gen_pnet_data import GenExamplePnet
from GenerateData.gen_tfrecords import GenRecordsPnet, GenRecordsRnet, GenRecordsOnet
from core.train import TrainPnet, TrainRnet, TrainOnet
from GenerateData.gen_hard_example import GenHardExampleRnet, GenHardExampleOnet
from GenerateData.gen_landmark import GetLandmarkPnet, GetLandmarkRnet, GetLandmarkOnet
import tensorflow as tf

print('start generating classification data for pnet')
GenExamplePnet()
print('start generating landmark data for pnet')
GetLandmarkPnet()
print('start writing tfrecords for pnet')
GenRecordsPnet()
print('start training pnet')
tf.reset_default_graph()
TrainPnet()



print('start generating classification data for rnet')
GenHardExampleRnet()
print('start generating landmark data for rnet')
GetLandmarkRnet()
print('start writing tfrecords for rnet')
GenRecordsRnet()
print('start training rnet')
tf.reset_default_graph()
TrainRnet()




print('start generating classification data for onet')
GenHardExampleOnet()
print('start generating landmark data for onet')
GetLandmarkOnet()
print('start writing tfrecords for onet')
GenRecordsOnet()
print('start training onet')
tf.reset_default_graph()
TrainOnet()










