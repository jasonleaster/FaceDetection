"""
Programmer  :   EOF
File        :   training.py
Date        :   2015.12.29
E-mail      :   jasonleaster@163.com

"""

from config import TRAINING_FACE
from config import TRAINING_NONFACE
from config import CASACADE_LIMIT

from cascade import Cascade

from time import time

from multiprocessing import  freeze_support

raise Exception("Unimplemented Cascade")

if __name__ == "__main__":
    freeze_support()

    start_time = time()
    model      = Cascade(TRAINING_FACE, TRAINING_NONFACE, limit = CASACADE_LIMIT)
    end_time   = time()

    print "total Cost time: ", end_time - start_time

    try:
        model.train()
        model.save()
    except KeyboardInterrupt:
        print "key board interrupt happened. training pause."
    
    

	


