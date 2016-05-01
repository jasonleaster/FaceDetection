"""
Programmer  :   EOF
E-mail      :   jasonleaster@gmail.com
File        :   mapReduce.py
Date        :   2016.04.15

File Description:

        This file contain two helpful function @Map and @Reduce
    which will help us to do parallel computing to accelerate the 
    process to compute features of images.

"""

from config     import PROCESS_NUM
from config     import FEATURE_FILE_SUBSET
from config     import TRAINING_IMG_WIDTH
from config     import TRAINING_IMG_HEIGHT

from mr_routine import routine
from haarFeature import Feature

from multiprocessing import Process
from image      import ImageSet

import numpy


def map(Face, NonFace):

    assert isinstance(Face,    ImageSet)
    assert isinstance(NonFace, ImageSet)

    # Multi-Process for acceleration
    images    = Face.images + NonFace.images
    images_num= len(images)
    processes = []

    for i in xrange(PROCESS_NUM):
        start = int((i    *1./PROCESS_NUM) * images_num)
        end   = int(((i+1)*1./PROCESS_NUM) * images_num )
        sub_imgs = images[start:end]

        process = Process(target = routine,
                            args = (sub_imgs,
                                    FEATURE_FILE_SUBSET + str(i) + ".cache")) 
        processes.append(process)
        
    for i in xrange(PROCESS_NUM):
        processes[i].start()

    for i in xrange(PROCESS_NUM):
        processes[i].join()


def reduce():
    from config import FEATURE_FILE_TRAINING
    from config import FEATURE_FILE_SUBSET
    from config import PROCESS_NUM

    mats = []
    tot_samples = 0
    for i in xrange(PROCESS_NUM):
        sub_mat = numpy.load(FEATURE_FILE_SUBSET + str(i) + ".cache" + ".npy")
        mats.append(sub_mat)
        tot_samples += sub_mat.shape[1]

    haar = Feature(TRAINING_IMG_WIDTH, TRAINING_IMG_HEIGHT)

    mat  = numpy.zeros((haar.featuresNum, tot_samples), numpy.float32)
    sample_readed = 0
    for i in xrange(PROCESS_NUM):
        for m in xrange(mats[i].shape[0]): # feature number
            for n in xrange(mats[i].shape[1]): # sample number

                mat[m][n + sample_readed] = mats[i][m][n]

        sample_readed += mats[i].shape[1]

    numpy.save(FEATURE_FILE_TRAINING, mat)

    return mat
