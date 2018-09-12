#coding:utf-8
import tensorflow as tf
import sys

sys.path.append('..')
from config import cfg

'''
label:
1	positive
0	negative
-1	part
-2	landmark

cls_prob:           shape=(384, 2)
label:              shape=(384, )
'''
def ComputeLossCls(cls_prob, label):
    #label=1 or label=0 then do classification
    zeros = tf.zeros_like(label)
   
    label_filter_invalid = tf.where(tf.less(label,0), zeros, label)     # (384, )
    num_cls_prob = tf.size(cls_prob)                                    # (1) = 384*2
    cls_prob_reshape = tf.reshape(cls_prob,[num_cls_prob,-1])           # (768, )
    label_int = tf.cast(label_filter_invalid,tf.int32)                  # (384, )
    num_row = tf.to_int32(cls_prob.get_shape()[0])                      # (1) = 384
    row = tf.range(num_row)*2                                           # (384, )
    indices_ = row + label_int                                          # (384, )
    
    # label_prob是(384, )的tensor， 到这里实际上是手动实现了softmax loss
    label_prob = tf.squeeze(tf.gather(cls_prob_reshape, indices_))      # (384, )
    loss = -tf.log(label_prob+1e-10)
    zeros = tf.zeros_like(label_prob, dtype=tf.float32)
    ones = tf.ones_like(label_prob,dtype=tf.float32)
    valid_inds = tf.where(label < zeros,zeros,ones)
    num_valid = tf.reduce_sum(valid_inds)
    keep_num = tf.cast(num_valid*cfg.hard_example_ratio,dtype=tf.int32)
    #set 0 to invalid sample
    loss = loss * valid_inds
    loss,_ = tf.nn.top_k(loss, k=keep_num)
    return tf.reduce_mean(loss)
    

'''
bbox_pred:          shape=(384, 4)
bbox_target:        shape=(384, 4)
label:              shape=(384, )
'''
def ComputeLossBbox(bbox_pred, bbox_target, label):
    #label=1 or label=-1 then do regression
    zeros_index = tf.zeros_like(label, dtype=tf.float32)                    # (384, )
    ones_index = tf.ones_like(label,dtype=tf.float32)                       # (384, )
    valid_inds = tf.where(tf.equal(tf.abs(label), 1),ones_index,zeros_index)# (384, )
    #(batch,)
    square_error = tf.square(bbox_pred-bbox_target)                         # (384, 4)
    square_error = tf.reduce_sum(square_error,axis=1)                       # (384， )
    #keep_num scalar
    num_valid = tf.reduce_sum(valid_inds)
    #keep_num = tf.cast(num_valid*cfg.hard_example_ratio,dtype=tf.int32)
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    #keep valid index square_error
    square_error = square_error*valid_inds
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    return tf.reduce_mean(square_error)

def ComputeLossLandmark(landmark_pred, landmark_target, label):
    #keep label =-2  then do landmark detection
    ones = tf.ones_like(label,dtype=tf.float32)
    zeros = tf.zeros_like(label,dtype=tf.float32)
    valid_inds = tf.where(tf.equal(label,-2),ones,zeros)
    square_error = tf.square(landmark_pred-landmark_target)
    square_error = tf.reduce_sum(square_error,axis=1)
    num_valid = tf.reduce_sum(valid_inds)
    #keep_num = tf.cast(num_valid*cfg.hard_example_ratio, dtype=tf.int32)
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    square_error = square_error*valid_inds
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    return tf.reduce_mean(square_error)
    
def ComputeAcc(cls_prob, label):
    pred = tf.argmax(cls_prob,axis=1)
    label_int = tf.cast(label,tf.int64)
    cond = tf.where(tf.greater_equal(label_int,0))
    picked = tf.squeeze(cond)
    label_picked = tf.gather(label_int,picked)
    pred_picked = tf.gather(pred,picked)
    accuracy_op = tf.reduce_mean(tf.cast(tf.equal(label_picked,pred_picked),tf.float32))
    return accuracy_op
    
def ComputeRecall(cls_prob, label):
    pred = tf.argmax(cls_prob,axis=1)
    label_int = tf.cast(label,tf.int64)
    cond = tf.where(tf.equal(label_int,1))
    picked = tf.squeeze(cond)
    label_picked = tf.gather(label_int,picked)
    pred_picked = tf.gather(pred,picked)
    recall_op = tf.reduce_mean(tf.cast(tf.equal(label_picked,pred_picked),tf.float32))
    return recall_op


            