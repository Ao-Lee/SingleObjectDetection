import tensorflow as tf

class FcnDetector(object):

    def __init__(self, net, model_path):
        graph = tf.Graph()
        with graph.as_default():
            self.image_op = tf.placeholder(tf.float32, name='input_image')
            self.width_op = tf.placeholder(tf.int32, name='image_width')
            self.height_op = tf.placeholder(tf.int32, name='image_height')
            image_reshape = tf.reshape(self.image_op, [1, self.height_op, self.width_op, 3])  
            self.prob, self.reg, _ = net(image_reshape, training=False)
            
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
            saver = tf.train.Saver()
            model_folder = '\\'.join(model_path.split('\\')[:-1])
            ckpt = tf.train.get_checkpoint_state(model_folder)
            assert ckpt and ckpt.model_checkpoint_path, "the model path is not valid"
            saver.restore(self.sess, model_path)
            
    def predict(self, batch):
        h, w, _ = batch.shape
        feed_dict={self.image_op:batch, self.width_op:w, self.height_op:h}
        prob, reg = self.sess.run([self.prob, self.reg], feed_dict=feed_dict)
        return prob, reg
