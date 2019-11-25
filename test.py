import tensorflow as tf
import numpy as np
from config import Config
from lstm import CLSTM_Cell
from log import Clog
from Data import CEletricDate as CNet_Input
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def build_network(config, inputs, lables, sequence_length):
    model = CLSTM_Cell(config)
    return model.forward(inputs, lables, sequence_length)


if __name__ == '__main__':
    config = Config
    log = Clog(config)
    #net_input为网络输入类，根据该类，每一次获取网络输入的数据
    net_input = CNet_Input(config)
    with tf.Session() as sess:
        '''
        定义网络输入
        inputs:[config.batchsize, None, config.input_dim]
        lables:[config.batchsize, 2, config.labels_num]
        sequence_length：定义一个训练序列的的长度。
        '''
        global_step = tf.Variable(0, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
        inputs = tf.placeholder(tf.float32,shape=[config.batch_size,None,config.input_dim])
        lables = tf.placeholder(tf.float32,shape=[config.batch_size,2,config.lables_num])
        sequence_length = tf.placeholder(tf.int32)
        
        #构建网络
        predicate_class, predicate_regress, loss = build_network(config, inputs, lables, sequence_length)
       
 
        #初始化训练网络变量
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        #网络参数保存
        saver = tf.train.Saver(max_to_keep=config.max_tp_keep)        
        checkpoint = tf.train.get_checkpoint_state('./results')
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
                
        #获取训练集样本数量
        Test_sample_num=net_input.test_sample_num
        for epoch in range(1):
            for step in range(Test_sample_num):
                #获取网络输入sequence和其对应lable
                test_input, test_lable = net_input.test_input()
                sequence_len = test_input.shape[1]
                #run
                run_item = sess.run([predicate_class, loss], feed_dict={inputs:test_input, lables:test_lable, sequence_length:sequence_len})
                if((step + 1) % config.log_interval == 0):
                    log.write("Sample:{}; predicates:{}; lables{}".format(step+1, run_item[0], test_lable[:,0,:]))
