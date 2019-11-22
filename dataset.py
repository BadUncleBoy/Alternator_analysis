import numpy as np
import random
class CNet_Input(object):
    '''
        this class is used to generate samples;labels;
        all the implemention is Inherited from this class
    '''
    def __init__(self,config):
        self.config = config
        self.train_sample_num = 100
        self.eval_sample_num = 10

    def train_input(self):
        
        '''
        through train_input method, get tht network's traing_sample and sample_lable 
        return value format
            train_sample :  [Batch_size, sequence_length, feature_size]
            lable        :  [Batch_size, one_hot_lable_num]
        notice :    sequence_length is not fixed
        '''
        return np.zeros((1,10,self.config.input_dim)),np.zeros((1,8))

    def val_input(self):
        
        '''
        through val_input method, get tht network's val_sample and val_lable 
        return value format
            val_sample :  [Batch_size, sequence_length, feature_size]
            lable        :  [Batch_size, one_hot_lable_num]
        notice :    sequence_length is not fixed
        '''
        pass
    
    def test_input(self):
        
        '''
        through test_input method, get tht network's test_sample 
        return value format
            sample :  [Batch_size, sequence_length, feature_size]
        notice :    sequence_length is not fixed
        '''
        pass
