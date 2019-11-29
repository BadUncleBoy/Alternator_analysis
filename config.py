import argparse
def parse_training_args(parser):
    parser.add_argument('--workdir',type=str, default='./',
                        help='the work dir of this program')
    
    parser.add_argument('--input_dim',type=int,default=2500,
                        help='def the input sampling length of the sinusoid')
    
    parser.add_argument('--batch_size',type=int,default=1)
     
    parser.add_argument('--hidden_dim',type=int,default=64)
    
    parser.add_argument('--model_dir',type=str,default='./results')
    
    parser.add_argument('--max_tp_keep',type=int,default=5,
                        help='only save referring number weights file')

    parser.add_argument('--epoch',type=int,default=100,
                        help='training epoches')

    parser.add_argument('--lables_num',type=int,default=3,
                        help='the num of the predicating lables')
   
    parser.add_argument('--eval_interval',type=int,default=100,
                        help='after training referring steps,conduct one evaluation')

    parser.add_argument('--save_ckpt_path',type=str,default='results/res.ckpt',
                        help='name and path to the weights file')
    
    #dataset dir
    parser.add_argument('--train_set_dir',type=str, default='data/dataset/train/')

    parser.add_argument('--eval_set_dir',type=str,default='data/dataset/eval/')

    parser.add_argument('--test_set_dir',type=str,default='data/dataset/train/')
    
    #logging parameter
    parser.add_argument('--log_interval',type=int,default=1,
                        help='output log after the referring training steps')

    parser.add_argument('--train_log_file',type=str,default='log/train_log.txt',
                    help='the train_log message to the inferring train_log file')
    
    parser.add_argument('--test_log_file',type=str,default='log/test_log.txt',
                    help='the test_log message to the inferring test_log file')

    parser.add_argument('--log_mode',type=str,default='s',
                    help='logging mode, s:output message to screen,f:output to file')
    
    #Adam parameter
    parser.add_argument('--optimizer',type=str,default='adam',
                    help='Using adam to train')
    
    parser.add_argument('--adam_beta1',type=float,default=0.9)

    parser.add_argument('--adam_beta2',type=float,default=0.999)

    parser.add_argument('--adam_epsilon',type=float,default=1e-08)
    
    #Exponential learing_rete parameter
    parser.add_argument('--learning_rate',type=float,default=0.1,
                    help='Initial learning rate')

    parser.add_argument('--learning_rate_decay_type',type=str,default='exponential',
                    help='Specifies how the learning rate is decayed. One of "fixed", "exponential",'
                    ' or "polynomial"')
    
    parser.add_argument('--decay_rate',type=float,default=0.9,
                    help='Learning rate decay factor')
    return parser.parse_args()

parser = argparse.ArgumentParser('config of the lstm network')
Config = parse_training_args(parser)
