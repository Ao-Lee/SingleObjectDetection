import tensorflow as tf
import numpy as np

class Detector(object):
    '''
    detectors[1] = Detector(R_Net, 24, 1, paths[1])
    detectors[2] = Detector(O_Net, 48, 1, paths[2])
    '''
    
    def __init__(self, net_factory, data_size, batch_size, model_path):
        graph = tf.Graph()
        with graph.as_default():
            self.image_op = tf.placeholder(tf.float32, shape=[batch_size, data_size, data_size, 3], name='input_image')
         
            self.prob, self.reg, self.landmark = net_factory(self.image_op, training=False)
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
            saver = tf.train.Saver()
            #check whether the dictionary is valid
            model_folder = '\\'.join(model_path.split('\\')[:-1])
            ckpt = tf.train.get_checkpoint_state(model_folder)
            assert ckpt and ckpt.model_checkpoint_path, "the model path is not valid"
            saver.restore(self.sess, model_path)

        self.data_size = data_size
        self.batch_size = batch_size
    #rnet and onet minibatch(test)
    def predict(self, databatch):

        batch_size = self.batch_size

        minibatch = []
        cur = 0
        #num of all_data
        n = databatch.shape[0]
        while cur < n:
            #split mini-batch
            minibatch.append(databatch[cur:min(cur + batch_size, n), :, :, :])
            cur += batch_size
        #every batch prediction result
        prob_list = []
        reg_list = []
        landmark_list = []
        for idx, data in enumerate(minibatch):
            m = data.shape[0]
            real_size = self.batch_size
            #the last batch 
            if m < batch_size:
                passed = np.arange(m)
                #gap (difference)
                gap = self.batch_size - m
                while gap >= len(passed):
                    gap -= len(passed)
                    passed = np.concatenate((passed, passed))
                if gap != 0:
                    passed = np.concatenate((passed, passed[:gap]))
                data = data[passed]
                real_size = m

            prob, reg, landmark = self.sess.run([self.prob, self.reg, self.landmark], feed_dict={self.image_op: data})
            prob_list.append(prob[:real_size])
            reg_list.append(reg[:real_size])
            landmark_list.append(landmark[:real_size])
            
        return np.concatenate(prob_list, axis=0), np.concatenate(reg_list, axis=0), np.concatenate(landmark_list, axis=0)
