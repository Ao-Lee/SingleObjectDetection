from os.path import join, exists
import os

debug = True
'''project path, change your path here'''
path_code = 'F:\\Dropbox\\DataScience\\Project\\MTCNN-TF-Vehicle'

'''path to detection dataset'''
path_detection = join(path_code, 'TrainingSets', 'detection')
path_detection_imgs = join(path_detection, 'imgs')
path_detection_labels = join(path_detection, 'anno_detection.txt')


'''path to intermediate outputs'''
path_output_files = 'E:\\DM\\MTCNN\\TMP' # requires large disk space in this folder
path_output_txt = join(path_output_files, 'txt')
path_output_models = join(path_output_files, 'models')
path_output_logs = join(path_output_files, 'logs')
if not exists(path_output_files): os.makedirs(path_output_files)
if not exists(path_output_models): os.makedirs(path_output_models)
if not exists(path_output_logs): os.makedirs(path_output_logs)
if not exists(path_output_txt): os.makedirs(path_output_txt)

'''path to landmark dataset'''
path_landmark = join(path_code, 'TrainingSets', 'landmark')
path_landmark_imgs = join(path_landmark, 'imgs')
path_landmark_labels = join(path_landmark, 'anno_landmark.txt')

'''general config'''
classtype = {'neg':0, 'pos':1, 'part':-1, 'landmark':-2}
resize = {'pnet':12, 'rnet':24, 'onet':48}

'''training params'''
# epoch = [18, 14, 22]
epoch = [14, 18, 22] # training epoch for pnet, rnet and onet
BATCH_SIZE = 350 # batch size during training
# weighted propotions for each data type in a merged training batch 
weights = {'pos':1, 'part':1, 'neg':3, 'landmark':0.01}
# learning rate decay policy
# for example, in this setting, use initial learning rate for the first 0.3 training process
# the learning rate will be decayed by a factor of 0.1**1 during 0.3~0.6 training process
# the learning rate will be decayed by a factgor of 0.1**3 after 0.9 training process
lr_decay_factor = 0.1
lr_decay_boundary = [0.3, 0.6, 0.9]

'''model params'''
hard_example_ratio = 0.7 # hard example selection ratio, only enabled in classification

# how much each loss contributes to total loss
loss_ratio = {
            'pnet': {'cls':1.0, 'bbox':0.5, 'landmark':0.0},
            'rnet': {'cls':1.0, 'bbox':0.5, 'landmark':0.0},
            'onet': {'cls':1.0, 'bbox':0.5, 'landmark':0.0}
             }
'''
loss_ratio = {
            'pnet': {'cls':1.0, 'bbox':0.5, 'landmark':0.5},
            'rnet': {'cls':1.0, 'bbox':0.5, 'landmark':0.5},
            'onet': {'cls':1.0, 'bbox':0.5, 'landmark':1.0}
             }
'''


