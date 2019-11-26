import tensorflow as tf
import numpy as np
from config import Config
from lstm import CLSTM_Cell
from log import Clog
from Data import CEletricDate as CNet_Input
from evaluate import evaluate_precision, evaluate_error_rate

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def build_network(config, inputs, lables, sequence_length):
    model = CLSTM_Cell(config)
    return model.forward(inputs, lables, sequence_length)


if __name__ == '__main__':
    config = Config
    log = Clog(config, is_train=False)
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
        checkpoint = tf.train.get_checkpoint_state(config.model_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
                
        #获取训练集样本数量
        Test_sample_num=net_input.test_sample_num
        Lables = np.zeros((Test_sample_num, 2, config.lables_num))
        Predicates = np.zeros((Test_sample_num, 2, config.lables_num))
        for step in range(Test_sample_num):
            #获取网络输入sequence和其对应lable
            test_input, test_lable = net_input.test_input()
            sequence_len = test_input.shape[1]
            #run
            run_item = sess.run([predicate_class, predicate_regress], feed_dict={inputs:test_input, lables:test_lable, sequence_length:sequence_len})
            log.write("Sample:{}; predicates_class :{}; lables{}".format(step+1, run_item[0], test_lable[:,0,:]))
            log.write("Sample:{}: predicates_regree:{}; lables{}".format(step+1, run_item[1], test_lable[:,1,:]))
              
            Lables[step] = test_lable[0]
            Predicates[step,0,:] = run_item[0][0]
            Predicates[step,1,:] = run_item[1][0]
        precision  = evaluate_precision(Predicates[:,0,:], Lables[:,0,:])
        error_rate = evaluate_error_rate(Predicates[:,1,:], Lables[:,0,:], Lables[:,1,:])
        log.write("Total predision: {}".format(precision))
        log.write("Total error_rate:{}".format(error_rate))
