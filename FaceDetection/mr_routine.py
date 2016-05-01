"""
Programmer  :   EOF
File        :   mr_routine.py
Date        :   2015.12.29
E-mail      :   jasonleaster@163.com

"""
from config     import TRAINING_IMG_WIDTH
from config     import TRAINING_IMG_HEIGHT

from haarFeature import Feature
import numpy
from functools import wraps
def processMeassure(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        import os
        import time
        print "process ", os.getpid(), "started!"
        start = time.time()
        fn(*args, **kwargs)
        end   = time.time()
        print "Cost time: ", end - start, " second."
        print "Process "   , os.getpid(), " end!"

    return measure_time

@processMeassure
def routine(images, filename):
    tot_samples = len(images)

    haar = Feature(TRAINING_IMG_WIDTH, TRAINING_IMG_HEIGHT)

    mat = numpy.zeros((haar.featuresNum, tot_samples), dtype = numpy.float32)

    for i in xrange(tot_samples):
        featureVec = haar.calFeatureForImg(images[i])
        for j in xrange(haar.featuresNum):
            mat[j][i]  = featureVec[j]

    numpy.save(filename, mat)
