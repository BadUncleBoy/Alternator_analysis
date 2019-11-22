import logging 
from config import Config
class Clog(object):
    def __init__(self,config=Config,is_train=True):
        '''
            this class is used to format show training process's loss situation
            the loss can be saved to a file or just print to the screen
        '''
        train_log_file = config.train_log_file
        test_log_file = config.test_log_file
        mode = config.log_mode
        
        self.logger = logging.getLogger('lstm_logger')
        self.logger.setLevel(logging.INFO)
        handler = None
        if(mode == 'f'):
            if(is_train):
                handler = logging.FileHandler(train_log_file)
            else:
                handler = logging.FileHandler(test_log_file)
        elif(mode == 's'):
            handler = logging.StreamHandler()
        else:
            raise ValueError('No defined loggin mode')
        handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s-%(name)s-{%(message)s}')
        
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def write(self, message):
        self.logger.info(message)