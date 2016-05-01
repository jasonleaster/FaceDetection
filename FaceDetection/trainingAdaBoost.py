"""
Programmer  :   EOF
File        :   trainingAdaBoost.py
Date        :   2015.12.29
E-mail      :   jasonleaster@163.com

"""

from config   import POSITIVE_SAMPLE
from config   import NEGATIVE_SAMPLE
from config   import TRAINING_IMG_HEIGHT
from config   import TRAINING_IMG_WIDTH
from config   import FEATURE_FILE_TRAINING
from config   import FEATURE_NUM
from config   import ADABOOST_LIMIT
from config   import ADABOOST_CACHE_FILE
from config   import DEBUG_MODEL
from config   import TRAINING_FACE
from config   import TRAINING_NONFACE

from haarFeature import Feature
from image       import ImageSet
from adaboost    import AdaBoost
from adaboost    import getCachedAdaBoost

import os
import numpy

Face    = ImageSet(TRAINING_FACE,    sampleNum = POSITIVE_SAMPLE)
nonFace = ImageSet(TRAINING_NONFACE, sampleNum = NEGATIVE_SAMPLE)

tot_samples = Face.sampleNum + nonFace.sampleNum

haar   = Feature(TRAINING_IMG_WIDTH, TRAINING_IMG_HEIGHT)

if os.path.isfile(FEATURE_FILE_TRAINING + ".npy"):

    _mat = numpy.load(FEATURE_FILE_TRAINING + ".npy")

else:
    if DEBUG_MODEL is True:
        _mat = numpy.zeros((haar.featuresNum, tot_samples))

        for i in xrange(Face.sampleNum):
            featureVec = haar.calFeatureForImg(Face.images[i])
            for j in xrange(haar.featuresNum):
                _mat[j][i                     ]  = featureVec[j]

        for i in xrange(nonFace.sampleNum):
            featureVec = haar.calFeatureForImg(nonFace.images[i])
            for j in xrange(haar.featuresNum):
                _mat[j][i + Face.sampleNum] = featureVec[j]

        numpy.save(FEATURE_FILE_TRAINING, _mat)
    else:
        from mapReduce import map
        from mapReduce import reduce

        map(Face, nonFace)
        _mat = reduce()

mat = _mat

featureNum, sampleNum = _mat.shape

assert sampleNum  == (POSITIVE_SAMPLE + NEGATIVE_SAMPLE)
assert featureNum == FEATURE_NUM

Label_Face    = [+1 for i in xrange(POSITIVE_SAMPLE)]
Label_NonFace = [-1 for i in xrange(NEGATIVE_SAMPLE)]

label = numpy.array(Label_Face + Label_NonFace)

cache_filename = ADABOOST_CACHE_FILE + str(0)

if os.path.isfile(cache_filename):
    model = getCachedAdaBoost(mat     = _mat,
                              label   = label,
                              filename= cache_filename,
                              limit   = ADABOOST_LIMIT)
else:
    model = AdaBoost(mat, label, limit = ADABOOST_LIMIT)
    model.train()
    model.saveModel(cache_filename)

print model
