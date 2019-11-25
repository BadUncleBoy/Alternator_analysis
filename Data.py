import os
import random
import numpy as np
from config import Config
from dataset import CNet_Input
class CEletricDate(CNet_Input):
    
    def __init__(self,conifg):
        self.config = conifg
        self.eval_sample_num  = 0
        self.test_sample_num  = 0
        
        self.index = 0
        self.test_index = 0
        
        self.lables_name,\
        self.train_sample_num,\
        self.train_sample_paths,\
        self.train_sample_error_kinds\
                =self.__parse_train_path()
        
        self.test_sample_num,\
        self.test_sample_paths,\
        self.test_sample_error_kinds\
                = self.__parse_test_path()
        
        
        self.__shuffle_data()

    def __shuffle_data(self):
        random_shuffle_list = list(range(self.train_sample_num))
        random.shuffle(random_shuffle_list)

        train_sample_paths = []
        train_sample_error_kinds = []
        
        for each in random_shuffle_list:
            train_sample_paths.append(self.train_sample_paths[each])
            train_sample_error_kinds.append(self.train_sample_error_kinds[each])

        self.train_sample_paths = train_sample_paths
        self.train_sample_error_kinds = train_sample_error_kinds
        
    def __parse_test_path(self):
        test_sample_paths = []
        test_sample_error_kinds = []
        test_sample_num = 0
        
        test_dir_path = os.path.join(self.config.workdir, self.config.test_set_dir)
        error_dir_names = os.listdir(test_dir_path)
        
        for error_dir_name in error_dir_names:
            test_error_dir_path = os.path.join(test_dir_path, error_dir_name + '/')
            for sample_name in os.listdir(test_error_dir_path):
                test_sample_paths.append(os.path.join(test_error_dir_path, sample_name))
                test_sample_error_kinds.append(self.lables_name.index(error_dir_name))
                test_sample_num += 1
        
        return test_sample_num, test_sample_paths, test_sample_error_kinds

    def __parse_train_path(self):
        train_sample_paths = []
        train_sample_error_kinds = []
        train_sample_num = 0
        train_dir_path = os.path.join(self.config.workdir, self.config.train_set_dir)
        error_dir_names = os.listdir(train_dir_path)
        
        for index, error_dir_name in enumerate(error_dir_names):
            train_error_dir_path = os.path.join(train_dir_path, error_dir_name + '/')
            for sample_name in os.listdir(train_error_dir_path):
                train_sample_paths.append(os.path.join(train_error_dir_path, sample_name))
                train_sample_error_kinds.append(index)
                train_sample_num += 1
        
        return error_dir_names, train_sample_num, train_sample_paths, train_sample_error_kinds
    
    def train_input(self):
        if(self.index == self.train_sample_num):
            self.__shuffle_data()
            self.index = 0
        
        input_data_file = self.train_sample_paths[self.index]
        input_data_kind = self.train_sample_error_kinds[self.index]
        input_sequence = None
        
        with open(input_data_file) as f:
            for line in f.readlines():
                input_sequence = line.strip().split(',')
                break
        length = len(input_sequence) // self.config.input_dim
        
        input_data = np.zeros((1, length, self.config.input_dim))
        input_label = np.zeros((1, 2, self.config.lables_num))
        
        input_sequence = [float(i) for i in input_sequence]
        for index in range(length):
            input_data[0,index,:] = input_sequence[index * self.config.input_dim:(index + 1) * self.config.input_dim]
        
        input_label[0, 0, input_data_kind] = 1
        
        self.index += 1

        return input_data, input_label        

    
    def test_input(self):
        input_data_file = self.test_sample_paths[self.test_index]
        input_data_kind = self.test_sample_error_kinds[self.test_index]
        input_sequence = None
        
        with open(input_data_file) as f:
            for line in f.readlines():
                input_sequence = line.strip().split(',')
                break
        length = len(input_sequence) // self.config.input_dim
        
        input_data = np.zeros((1, length, self.config.input_dim))
        input_label = np.zeros((1, 2, self.config.lables_num))
        
        input_sequence = [float(i) for i in input_sequence]
        for index in range(length):
            input_data[0,index,:] = input_sequence[index * self.config.input_dim:(index + 1) * self.config.input_dim]
        
        input_label[0, 0, input_data_kind] = 1
        
        self.test_index += 1
        
        return input_data, input_label

if __name__ == '__main__':
    data = CEletricDate(Config)
    
    