from config import Config
from log import Clog
import random
log = Clog(Config)
num=1000
for index in range(num):
    random_value = random.randint(0,8)
    log.write("sample:{}'ture_class:{};predicate_class:{}".format(index,random_value,random_value))
    log.write("sample:{};ture_regree:{};predicate_regree:{}".format(index,(8+2*random.random())*10,(5+5*random.random())*10))
