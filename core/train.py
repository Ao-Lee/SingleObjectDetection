#coding:utf-8
import tensorflow as tf
import numpy as np
import os
from os.path import join, exists
import sys
import random
import cv2

try:
    from .model import P_Net, R_Net, O_Net
    from .read_tfrecord import read_multi_tfrecords
except ImportError:
    from model import P_Net, R_Net, O_Net
    from read_tfrecord import read_multi_tfrecords

sys.path.append('..')
from config import cfg

def GetBatches():
    weights = {k:v/sum(list(cfg.weights.values())) for k, v in cfg.weights.items()}
    batches = {k:max(1, np.floor(v*cfg.BATCH_SIZE)) for k,v in weights.items()}
    batches = {k:int(v) for k,v in batches.items()}
    batches['landmark'] = batches['landmark'] + cfg.BATCH_SIZE - sum(list(batches.values())) 
    assert cfg.BATCH_SIZE == sum(list(batches.values()))
    return batches

def GetTrainigOp(base_lr, total_loss, datasize, total_epoch):
    total_step = int(datasize / cfg.BATCH_SIZE * total_epoch)
    boundaries = [int(total_step*boundary) for boundary in cfg.lr_decay_boundary]
    lr_values = [base_lr * (cfg.lr_decay_factor**b) for b in range(0, len(cfg.lr_decay_boundary) + 1)]
    global_step = tf.Variable(0, trainable=False)
    lr_op = tf.train.piecewise_constant(global_step, boundaries, lr_values)
    optimizer = tf.train.MomentumOptimizer(lr_op, 0.9)
    train_op = optimizer.minimize(total_loss, global_step)
    return train_op, lr_op

# all mini-batch mirror
def random_flip_images(image_batch,label_batch,landmark_batch):
    #mirror
    if random.choice([0,1]) > 0:
        # num_images = image_batch.shape[0]
        fliplandmarkindexes = np.where(label_batch==-2)[0]
        flipposindexes = np.where(label_batch==1)[0]
        #only flip
        flipindexes = np.concatenate((fliplandmarkindexes,flipposindexes))
        #random flip    
        for i in flipindexes:
            cv2.flip(image_batch[i],1,image_batch[i])        
        
        #pay attention: flip landmark    
        for i in fliplandmarkindexes:
            landmark_ = landmark_batch[i].reshape((-1,2))
            landmark_ = np.asarray([(1-x, y) for (x, y) in landmark_])
            landmark_[[0, 1]] = landmark_[[1, 0]]#left eye<->right eye
            landmark_[[3, 4]] = landmark_[[4, 3]]#left mouth<->right mouth        
            landmark_batch[i] = landmark_.ravel()
        
    return image_batch,landmark_batch

def GetTrainingDataSize(target):
    assert target in ['pnet', 'onet', 'rnet']
    if cfg.debug: return 1000
    
    with open(join(cfg.path_output_txt, target + '_pos.txt'), 'r') as f:
        pos = f.readlines()
    with open(join(cfg.path_output_txt, target + '_neg.txt'), 'r') as f:
        neg = f.readlines()
    with open(join(cfg.path_output_txt, target + '_part.txt'), 'r') as f:
        part = f.readlines()
    '''
    with open(join(cfg.path_output_txt, target + '_landmark.txt'), 'r') as f:
        landmark = f.readlines()
    '''
    # size = len(pos) + len(neg) + len(part) + len(landmark)
    size = (len(pos) + len(neg) + len(part))
    return size
    
def SaveModel(session, saver, target, global_step):
    path = join(cfg.path_output_models, target)
    if not exists(path): os.makedirs(path)
    path = join(path, target)
    saver.save(session, path, global_step=global_step)

def train(net_factory, total_epoch=20, target=None, display=200, base_lr=0.01):
   
    assert target in ['pnet', 'onet', 'rnet']
    image_size = cfg.resize[target]
    
    pos_dir = join(cfg.path_output_files, '%s_pos.tfrecord'%target)
    part_dir = join(cfg.path_output_files, '%s_part.tfrecord'%target)
    neg_dir = join(cfg.path_output_files, '%s_neg.tfrecord'%target)
    landmark_dir = join(cfg.path_output_files, '%s_landmark.tfrecord'%target)
    dataset_dirs = [pos_dir,part_dir,neg_dir,landmark_dir]
    
    batches = GetBatches()
    batch_sizes = [batches['pos'], batches['part'], batches['neg'], batches['landmark']]
    image_batch, label_batch, bbox_batch,landmark_batch = read_multi_tfrecords(dataset_dirs,batch_sizes, image_size)        
    
    
    input_image = tf.placeholder(tf.float32, shape=[cfg.BATCH_SIZE, image_size, image_size, 3], name='input_image')
    label = tf.placeholder(tf.float32, shape=[cfg.BATCH_SIZE], name='label')
    bbox_target = tf.placeholder(tf.float32, shape=[cfg.BATCH_SIZE, 4], name='bbox_target')
    landmark_target = tf.placeholder(tf.float32,shape=[cfg.BATCH_SIZE, 10],name='landmark_target')
    
    
    cls_loss_op,bbox_loss_op,landmark_loss_op,L2_loss_op,accuracy_op, recall_op = net_factory(input_image, label, bbox_target,landmark_target, training=True)
    total_loss_op = cfg.loss_ratio[target]['cls']*cls_loss_op + cfg.loss_ratio[target]['bbox']*bbox_loss_op + cfg.loss_ratio[target]['landmark']*landmark_loss_op + L2_loss_op
    
    datasize = GetTrainingDataSize(target)
    train_op, lr_op = GetTrainigOp(base_lr, total_loss_op, datasize, total_epoch)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=0)
    sess.run(init)
    tf.summary.scalar("cls_loss",cls_loss_op)
    tf.summary.scalar("bbox_loss",bbox_loss_op)
    tf.summary.scalar("landmark_loss",landmark_loss_op)
    tf.summary.scalar("cls_accuracy",accuracy_op)
    tf.summary.scalar("cls_recall",recall_op)
    
    summary_op = tf.summary.merge_all()
    
    logs_dir = join(cfg.path_output_logs, target)
    if not os.path.exists(logs_dir): os.makedirs(logs_dir)
    writer = tf.summary.FileWriter(logs_dir,sess.graph)
    #begin 
    coord = tf.train.Coordinator()
    #begin enqueue thread
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    i = 0
    #total steps
    max_step = int(datasize / cfg.BATCH_SIZE + 1) * total_epoch
    print('total step for training {} is {}'.format(target, max_step))
    epoch = 0
    sess.graph.finalize() 
    try:
        for step in range(max_step):
            i = i + 1

            if coord.should_stop():
                break
            image_batch_array, label_batch_array, bbox_batch_array,landmark_batch_array = sess.run([image_batch, label_batch, bbox_batch,landmark_batch])
            image_batch_array,landmark_batch_array = random_flip_images(image_batch_array,label_batch_array,landmark_batch_array)
            _,_,summary = sess.run([train_op, lr_op ,summary_op], feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array,landmark_target:landmark_batch_array})
            
            if step % display == 0:
                operations = [cls_loss_op, bbox_loss_op,landmark_loss_op,L2_loss_op,lr_op,accuracy_op, recall_op]
                feed_dict = {input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array, landmark_target: landmark_batch_array}
                cls_loss, bbox_loss, landmark_loss,L2_loss,lr,acc, recall = sess.run(operations, feed_dict=feed_dict)

                print('Step: %d, acc: %2f, recall: %2f, closs: %2f, bloss: %2f, lr:%f ' % (step, acc, recall, cls_loss, bbox_loss, lr))

            if i * cfg.BATCH_SIZE > datasize*2:
                epoch = epoch + 1
                i = 0
                SaveModel(sess, saver, target, global_step=epoch*2)
            writer.add_summary(summary,global_step=step)
    except tf.errors.OutOfRangeError:
        print("完成！！！")
    finally:
        coord.request_stop()
        writer.close()
    coord.join(threads)
    sess.close()
      
def TrainPnet():
    train(P_Net, total_epoch=cfg.epoch[0], display=400, target='pnet', base_lr=0.01)
    
def TrainRnet():
    train(R_Net, total_epoch=cfg.epoch[1], display=400, target='rnet', base_lr=0.01)
    
def TrainOnet():
    train(O_Net, total_epoch=cfg.epoch[2], display=400, target='onet', base_lr=0.01)

if __name__=='__main__':
    pass
    # TrainPnet()
    # TrainRnet()
    # TrainOnet()
