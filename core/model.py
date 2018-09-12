import tensorflow as tf
from tensorflow.contrib import slim

try:
    from .loss import ComputeLossCls, ComputeLossBbox, ComputeLossLandmark
    from .loss import ComputeAcc, ComputeRecall
except ImportError:
    from loss import ComputeLossCls, ComputeLossBbox, ComputeLossLandmark
    from loss import ComputeAcc, ComputeRecall
    
'''
inputs:             shape=(384, 12, 12, 3)
label:              shape=(384, )
bbox_target:        shape=(384, 4)
landmark_target:    shape=(384, 10)
'''
def P_Net(inputs, label=None, bbox_target=None, landmark_target=None, training=False):
    with slim.arg_scope([slim.conv2d],
                        # activation_fn=prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005), 
                        padding='valid'):
        # print(inputs.get_shape())
        net = slim.conv2d(inputs, 10, 3, stride=1,scope='conv1')
        # print(net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[2,2], stride=2, scope='pool1', padding='SAME')
        # print(net.get_shape())
        net = slim.conv2d(net,num_outputs=16,kernel_size=[3,3],stride=1,scope='conv2')
        # print(net.get_shape())
        net = slim.conv2d(net,num_outputs=32,kernel_size=[3,3],stride=1,scope='conv3')
        # print(net.get_shape())
        #batch*H*W*2
        conv4_1 = slim.conv2d(net,num_outputs=2,kernel_size=[1,1],stride=1,scope='conv4_1',activation_fn=tf.nn.softmax)
        #conv4_1 = slim.conv2d(net,num_outputs=1,kernel_size=[1,1],stride=1,scope='conv4_1',activation_fn=tf.nn.sigmoid)
        
        # print(conv4_1.get_shape())          #(384, H, W, 2)
        #batch*H*W*4
        bbox_pred = slim.conv2d(net,num_outputs=4,kernel_size=[1,1],stride=1,scope='conv4_2',activation_fn=None)
        # print(bbox_pred.get_shape())        #(384, H, W, 4)
        #batch*H*W*10
        landmark_pred = slim.conv2d(net,num_outputs=10,kernel_size=[1,1],stride=1,scope='conv4_3',activation_fn=None)
        # print(landmark_pred.get_shape())    #(384, H, W, 10)
        #cls_prob_original = conv4_1 
        #bbox_pred_original = bbox_pred
        if training:
            #batch*2
            cls_prob = tf.squeeze(conv4_1,[1,2],name='cls_prob') #(384, 2)
            cls_loss = ComputeLossCls(cls_prob,label)
            #batch
            bbox_pred = tf.squeeze(bbox_pred,[1,2],name='bbox_pred')
            bbox_loss = ComputeLossBbox(bbox_pred,bbox_target,label)
            #batch*10
            landmark_pred = tf.squeeze(landmark_pred,[1,2],name="landmark_pred")
            landmark_loss = ComputeLossLandmark(landmark_pred,landmark_target,label)

            accuracy = ComputeAcc(cls_prob,label)
            recall = ComputeRecall(cls_prob,label)
            L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy,recall
        #test
        else:
            #when test,batch_size = 1
            
            cls_pro_test = tf.squeeze(conv4_1, axis=0)              # (1, 1, 2)
            bbox_pred_test = tf.squeeze(bbox_pred,axis=0)           # (1, 1, 4)
            landmark_pred_test = tf.squeeze(landmark_pred,axis=0)   # (1, 1, 10)
            return cls_pro_test,bbox_pred_test,landmark_pred_test
        
def R_Net(inputs, label=None, bbox_target=None, landmark_target=None, training=False):
    with slim.arg_scope([slim.conv2d],
                        # activation_fn = prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),                        
                        padding='valid'):
        # print(inputs.get_shape())
        net = slim.conv2d(inputs, num_outputs=28, kernel_size=[3,3], stride=1, scope="conv1")
        # print(net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool1", padding='SAME')
        # print(net.get_shape())
        net = slim.conv2d(net,num_outputs=48,kernel_size=[3,3],stride=1,scope="conv2")
        # print(net.get_shape())
        net = slim.max_pool2d(net,kernel_size=[3,3],stride=2,scope="pool2")
        # print(net.get_shape())
        net = slim.conv2d(net,num_outputs=64,kernel_size=[2,2],stride=1,scope="conv3")
        # print(net.get_shape())
        fc_flatten = slim.flatten(net)
        # print('fc_flatten: {}'.format(fc_flatten.get_shape()))
        # fc1 = slim.fully_connected(fc_flatten, num_outputs=128,scope="fc1", activation_fn=prelu)
        fc1 = slim.fully_connected(fc_flatten, num_outputs=128,scope="fc1")
        # print('fc1: {}'.format(fc1.get_shape()))
        #batch*2
        cls_prob = slim.fully_connected(fc1,num_outputs=2,scope="cls_fc",activation_fn=tf.nn.softmax)
        # print('cls_prob: {}'.format(cls_prob.get_shape()))
        #batch*4
        bbox_pred = slim.fully_connected(fc1,num_outputs=4,scope="bbox_fc",activation_fn=None)
        # print('bbox_pred: {}'.format(bbox_pred.get_shape()))
        #batch*10
        landmark_pred = slim.fully_connected(fc1,num_outputs=10,scope="landmark_fc",activation_fn=None)
        # print('landmark_pred: {}'.format(landmark_pred.get_shape()))
        #train
        if training:
            cls_loss = ComputeLossCls(cls_prob, label)
            bbox_loss = ComputeLossBbox(bbox_pred, bbox_target, label)
            accuracy = ComputeAcc(cls_prob, label)
            recall = ComputeRecall(cls_prob, label)
            landmark_loss = ComputeLossLandmark(landmark_pred, landmark_target, label)
            L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy, recall
        else:
            return cls_prob,bbox_pred,landmark_pred
    
def O_Net(inputs,label=None, bbox_target=None, landmark_target=None, training=False):
    with slim.arg_scope([slim.conv2d],
                        # activation_fn = prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),                        
                        padding='valid'):
        # print(inputs.get_shape())
        net = slim.conv2d(inputs, num_outputs=32, kernel_size=[3,3], stride=1, scope="conv1")
        # print(net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool1", padding='SAME')
        # print(net.get_shape())
        net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope="conv2")
        # print(net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool2")
        # print(net.get_shape())
        net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope="conv3")
        # print(net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, scope="pool3", padding='SAME')
        # print(net.get_shape())
        net = slim.conv2d(net,num_outputs=128,kernel_size=[2,2],stride=1,scope="conv4")
        # print(net.get_shape())              
        fc_flatten = slim.flatten(net)
        # print('fc_flatten: {}'.format(fc_flatten.get_shape()))
        # fc1 = slim.fully_connected(fc_flatten, num_outputs=256,scope="fc1", activation_fn=prelu)
        fc1 = slim.fully_connected(fc_flatten, num_outputs=256,scope="fc1")
        # print('fc1: {}'.format(fc1.get_shape()))
        #batch*2
        cls_prob = slim.fully_connected(fc1,num_outputs=2,scope="cls_fc",activation_fn=tf.nn.softmax)
        # print('cls_prob: {}'.format(cls_prob.get_shape()))
        #batch*4
        bbox_pred = slim.fully_connected(fc1,num_outputs=4,scope="bbox_fc",activation_fn=None)
        # print('bbox_pred: {}'.format(bbox_pred.get_shape()))
        #batch*10
        landmark_pred = slim.fully_connected(fc1,num_outputs=10,scope="landmark_fc",activation_fn=None)
        # print('landmark_pred: {}'.format(landmark_pred.get_shape()))     
        #train
        if training:
            cls_loss = ComputeLossCls(cls_prob, label)
            bbox_loss = ComputeLossBbox(bbox_pred,bbox_target, label)
            accuracy = ComputeAcc(cls_prob, label)
            recall = ComputeRecall(cls_prob, label)
            landmark_loss = ComputeLossLandmark(landmark_pred, landmark_target, label)
            L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy, recall
        else:
            return cls_prob,bbox_pred,landmark_pred