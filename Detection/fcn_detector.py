import tensorflow as tf
# import sys
# sys.path.append("../")


class FcnDetector(object):
    #net_factory: which net
    #model_path: where the params'file is
    def __init__(self, net_factory, model_path):
        #create a graph
        graph = tf.Graph()
        with graph.as_default():
            #define tensor and op in graph(-1,1)
            self.image_op = tf.placeholder(tf.float32, name='input_image')
            self.width_op = tf.placeholder(tf.int32, name='image_width')
            self.height_op = tf.placeholder(tf.int32, name='image_height')
            image_reshape = tf.reshape(self.image_op, [1, self.height_op, self.width_op, 3])
            
            '''
            cls_prob    (1, 1, 2)
            bbox_pred   (1, 1, 4)
            '''
            self.cls_prob, self.bbox_pred, _ = net_factory(image_reshape, training=False)
            
            #allow 
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
            saver = tf.train.Saver()
            #check whether the dictionary is valid
            model_folder = '\\'.join(model_path.split('\\')[:-1])
            ckpt = tf.train.get_checkpoint_state(model_folder)
            assert ckpt and ckpt.model_checkpoint_path, "the model path is not valid"
            saver.restore(self.sess, model_path)
    def predict(self, databatch):
        height, width, _ = databatch.shape
        # print(height, width)
        cls_prob, bbox_pred = self.sess.run([self.cls_prob, self.bbox_pred],
                                                           feed_dict={self.image_op: databatch, self.width_op: width,
                                                                      self.height_op: height})
        '''
        cls_prob    (1, 1, 2)
        bbox_pred   (1, 1, 4)
        '''
        return cls_prob, bbox_pred
