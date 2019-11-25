# Alternator_analysis
### This project is a commercial project v1,using generator current waveform to predict generator gault type and remaining working time.
## part1. Paralycite
1. Config.py
```bashrc
work_dir:   the work dir of this program
input_dim:  def the input sampling length of the sinusoid
hidden_dim: def the LSTM cell hidden dimension
model_dir:  def the dir to place the ckpt file
epoch:      def the training epoches
lables_num: def the num of error 
log_mode:   s:output message to screen f:output to file
log_dir:    
```
2. log.py
```bashrc
this file defines the format of the log, it can be shown on the screen and can also output to a file
when output to file, just use the config --log_mode 'f'
the default --log_mode is 's'
```
3. lstm.py
```bashrc
the lstm network
```
4. train.py
```bashrc
定义网络输入：
inputs:[config.batchsize, None, config.input_dim]
lables:[config.batchsize, 2, config.labels_num]
sequence_length：定义一个训练序列的的长度。
其中lables[:,0,:] 存储了故障类型的标签
lables[:,1,:] 存储了故障停用时间的标签
```
5. dataset.py
```bashrc
network's training data input，just a interface, it defines the 
format of the net's input.
train_input():函数生成训练网络所需要的数据
test_input(): 函数生成测试网络所需要的数据
```

## part2. requirements
```bashrc
tensorflow
numpy
```
## part3. Running
```bashrc
$ python train.py
此函数何以直接运行，在网络中有50个样本用于训练，训练网络输出结果存放在log/train_log.txt
```
```bashrc
$ python test.py
此函数可以直接运行，网络中有10个测试样本，测试网络输出结果存放在log/test_log.txt
```
## part4. Running
自己实现一个dataset的数据接口，其需继承自CNet_Input类。
在自定义类中，需要实现
```bashrc
def train_input(self):
    返回值: train_sample, train_lable
    返回值格式: train_sample [config.batchsize,sequence_length config.input_dim]
                train_lable [config.batchsize, 2, config.lables_num]
    返回格式说明: sequence_length: 一个序列的长度，LSTM网络可以时序输入
                train_lable[:,0,:]:存储发电机故障类型，one-hot编码
                train_lable[:,1,:]:发电机故障类型对应位置，存储的其因故障而停用时间。
def val_input(self):
    同上
def test_input(self):
    同上
三个函数。
还需其中定义三个变量:
train_sample_num:  训练集样本数量
test_sample_num:   验证集样本数量
eval_sample_num:   测试集样本数量
```