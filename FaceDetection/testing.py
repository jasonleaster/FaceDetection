from config import TEST_FACE
from config import TEST_NONFACE
from config import TRAINING_IMG_HEIGHT
from config import TRAINING_IMG_WIDTH
from config import ADABOOST_CACHE_FILE
from config import POSITIVE_SAMPLE
from config import LABEL_POSITIVE

from adaboost import getCachedAdaBoost

from image  import ImageSet
from haarFeature import Feature

import numpy

face    = ImageSet(TEST_FACE,    sampleNum = 100)

nonFace = ImageSet(TEST_NONFACE, sampleNum = 100)

tot_samples = face.sampleNum + nonFace.sampleNum

haar   = Feature(TRAINING_IMG_WIDTH, TRAINING_IMG_HEIGHT)

mat = numpy.zeros((haar.featuresNum, tot_samples))

for i in xrange(face.sampleNum):
    featureVec = haar.calFeatureForImg(face.images[i])
    for j in xrange(haar.featuresNum):
        mat[j][i                     ]  = featureVec[j]
        
for i in xrange(nonFace.sampleNum):
    featureVec = haar.calFeatureForImg(nonFace.images[i])
    for j in xrange(haar.featuresNum):
        mat[j][i + face.sampleNum] = featureVec[j]


model = getCachedAdaBoost(filename = ADABOOST_CACHE_FILE + str(0), limit = 10)

output = model.prediction(mat, th=0)

detectionRate = numpy.count_nonzero(output[0:100] == LABEL_POSITIVE) * 1./ 100

print output



