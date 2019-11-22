import tensorflow as tf
import numpy as np
from config import Config
from lstm import CLSTM_Cell
from log import Clog
from dataset import CNet_Input
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
def _config_optimizer(config, global_step, loss):
    learning_rate = None
    if(config.learning_rate_decay_type == 'exponential'):
        learning_rate = tf.train.exponential_decay(config.learning_rate,
                                          global_step=global_step,
                                          decay_steps=100,
                                          decay_rate=config.decay_rate,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    else:
        raise ValueError('No defined learning reate decay type')
    
    if(config.optimizer == 'adam'):
        optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate,
                beta1=config.adam_beta1,
                beta2=config.adam_beta2,
                epsilon=config.adam_epsilon,
                use_locking=False,
                name='Adam_optimizer')
        return optimizer.minimize(loss,global_step=global_step)
    else:
        raise ValueError('No defined optimizer type')

def build_network(config, inputs, lables, sequence_length):
    model = CLSTM_Cell(config)
    return model.forward(inputs, lables, sequence_length)

def evaluate(sess, net_input, inputs, lebles, loss, sequence_length, log):
    Eval_sample_num=net_input.eval_sample_num
    
    for step in range(Eval_sample_num):
        eval_input, eval_lable = net_input.val_input()
        sequence_len = eval_input.shape[1]
        run_item = sess.run([loss], feed_dict={inputs:eval_input, lables:eval_lable, sequence_length:sequence_len})

        if((step + 1) % config.log_interval == 0):
                    log.write("eval:step{},train_loss: {}".format(step, run_item[0]))

if __name__ == '__main__':
    config = Config
    log = Clog(config)
    net_input = CNet_Input(config)
    with tf.Session() as sess:
        global_step = tf.Variable(0, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
        inputs = tf.placeholder(tf.float32,shape=[config.batch_size,None,config.input_dim])
        lables = tf.placeholder(tf.float32,shape=[config.batch_size,8])
        sequence_length = tf.placeholder(tf.int32)
        net_output, loss, _ = build_network(config, inputs, lables, sequence_length)
        train_op = _config_optimizer(config, global_step, loss) 
        
        saver = tf.train.Saver(max_to_keep=config.max_tp_keep)
        
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        
        Train_sample_num=net_input.train_sample_num
        steps = 0
        for epoch in range(config.epoch):
            for step in range(Train_sample_num):
                train_input, train_lable = net_input.train_input()
                sequence_len = train_input.shape[1]
                run_item = sess.run([train_op, loss,net_output], feed_dict={inputs:train_input, lables:train_lable, sequence_length:sequence_len})
                steps += 1
                if((steps + 1) % config.log_interval == 0):
                    log.write("train:epoch{},step{},train_loss: {}".format(epoch+1, steps+1, run_item[1]))
                '''
                if((steps + 1) % config.eval_interval == 0):
                    evaluate(sess, net_input, inputs, lables, loss, sequence_length, log)
                '''
            saver.save(sess, save_path=config.save_ckpt_path, global_step=steps+1)
