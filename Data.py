import os
import random
import numpy as np
from dataset import CNet_Input
class CEletricDate(CNet_Input):
    
    def __init__(self,conifg):
        self.config = conifg
        self.train_sample_num = self.__get_train_sample_num()
        self.eval_sample_num  = self.__get_eval_sample_num()
        self.test_sample_num  = self.__get_test_sample_num()
        self.index = 0
        self.train_sample_paths, self.train_sample_error_kinds = self.__parse_train_path()
    
    def __get_train_sample_num(self):
        train_sample_num = 0
        train_dir_path = os.path.join(self.config.workdir, self.config.train_set_dir)
        error_dir_names = os.listdir(train_dir_path)
        for error_dir_name in error_dir_names:
            train_error_dir_path = os.path.join(train_dir_path, error_dir_name)
            for _ in os.listdir(train_error_dir_path):
                train_sample_num += 1
        return train_sample_num

    def __get_eval_sample_num(self):
        eval_sample_num = 0
        eval_dir_path = os.path.join(self.config.workdir, self.config.eval_set_dir)
        error_dir_names = os.listdir(eval_dir_path)
        for error_dir_name in error_dir_names:
            eval_error_dir_path = os.path.join(eval_dir_path, error_dir_name)
            for _ in os.listdir(eval_error_dir_path):
                eval_sample_num += 1
        return eval_sample_num

    def __get_test_sample_num(self):
        test_sample_num = 0
        test_dir_path = os.path.join(self.config.workdir, self.config.test_set_dir)
        error_dir_names = os.listdir(test_dir_path)
        for error_dir_name in error_dir_names:
            test_error_dir_path = os.path.join(test_dir_path, error_dir_name)
            for _ in os.listdir(test_error_dir_path):
                test_sample_num += 1
        return test_sample_num

    def __parse_train_path(self):
        train_sample_paths = []
        train_sample_error_kinds = []
        train_dir_path = os.path.join(self.config.workdir, self.config.train_set_dir)
        error_dir_names = os.listdir(train_dir_path)
        for index, error_dir_name in enumerate(error_dir_names):
            train_error_dir_path = os.path.join(train_dir_path, error_dir_name)
            for sample_name in os.listdir(train_error_dir_path):
                train_sample_paths.append(os.path.join(train_error_dir_path, sample_name))
                train_sample_error_kinds.append(index)
        return train_sample_paths, train_sample_error_kinds
    
    def __parse_eval_path(self):
        eval_sample_paths = []
        eval_sample_error_kinds = []
        eval_dir_path = os.path.join(self.config.workdir, self.config.eval_set_dir)
        error_dir_names = os.listdir(eval_dir_path)
        for index, error_dir_name in enumerate(error_dir_names):
            eval_error_dir_path = os.path.join(eval_dir_path, error_dir_name)
            for sample_name in os.listdir(eval_error_dir_path):
                eval_sample_paths.append(os.path.join(eval_error_dir_path, sample_name))
                eval_sample_error_kinds.append(index)
        return eval_sample_paths, eval_sample_error_kinds

    def train_input(self):
        
        if(self.index == self.train_sample_num):
            random_shuffle_list = list(range(self.test_sample_num))
            random.shuffle(random_shuffle_list)
            self.train_sample_paths = self.train_sample_paths[random_shuffle_list]
            self.train_sample_error_kinds = self.train_sample_error_kinds[random_shuffle_list]
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
        input_label = np.zeros((1,self.config.lables_num))
        
        input_sequence = [float(i) for i in input_sequence]
        for index in range(length):
            input_data[0,index,:] = input_sequence[index * self.config.input_dim:(index + 1) * self.config.input_dim]

        input_label[input_data_kind] = 1

        return input_data, input_label        


