"""
Programmer  :   EOF
File        :   cascade.py
Date        :   2016.01.17
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

from haarFeature import Feature
from image       import ImageSet
from adaboost    import AdaBoost
from adaboost    import getCachedAdaBoost

import os
import numpy


class Cascade:

    def __init__(self, face_dir = "", nonface_dir = "", train = True, limit = 3):
        #tot_samples = 0

        self.Face    = ImageSet(face_dir,    sampleNum = POSITIVE_SAMPLE)
        self.nonFace = ImageSet(nonface_dir, sampleNum = NEGATIVE_SAMPLE)

        tot_samples = self.Face.sampleNum + self.nonFace.sampleNum

        self.classifier = AdaBoost

        self.haar   = Feature(TRAINING_IMG_WIDTH, TRAINING_IMG_HEIGHT)

        if os.path.isfile(FEATURE_FILE_TRAINING + ".npy"):

            self._mat = numpy.load(FEATURE_FILE_TRAINING + ".npy")

        else:
            if DEBUG_MODEL is True:
                self._mat = numpy.zeros((self.haar.featuresNum, tot_samples))


                for i in xrange(self.Face.sampleNum):
                    featureVec = self.haar.calFeatureForImg(self.Face.images[i])
                    for j in xrange(self.haar.featuresNum):
                        self._mat[j][i                     ]  = featureVec[j]

                for i in xrange(self.nonFace.sampleNum):
                    featureVec = self.haar.calFeatureForImg(self.nonFace.images[i])
                    for j in xrange(self.haar.featuresNum):
                        self._mat[j][i + self.Face.sampleNum] = featureVec[j]

                numpy.save(FEATURE_FILE_TRAINING, self._mat)
            else:
                from mapReduce import map
                from mapReduce import reduce

                map(self.Face, self.nonFace)
                self._mat = reduce()



        featureNum, sampleNum = self._mat.shape

        assert sampleNum  == (POSITIVE_SAMPLE + NEGATIVE_SAMPLE)
        assert featureNum == FEATURE_NUM

        Label_Face    = [+1 for i in xrange(POSITIVE_SAMPLE)]
        Label_NonFace = [-1 for i in xrange(NEGATIVE_SAMPLE)]

        self._label = numpy.array(Label_Face + Label_NonFace)
        self.limit  = limit
        self.classifierNum     = 0
        self.strong_classifier = [None for i in xrange(limit)]


    def train(self):
        FPR = 1.0
        detection_rate = 0
        #while FPR > 0.1 and self.classifierNum < self.limit:
        for i in xrange(self.limit):

            cache_filename = ADABOOST_CACHE_FILE + str(i)

            if os.path.isfile(cache_filename):
                self.strong_classifier[i] = getCachedAdaBoost(mat     = self._mat,
                                                              label   = self._label,
                                                              filename= cache_filename,
                                                              limit   = 30)
            else:
                self.strong_classifier[i] = AdaBoost(self._mat, self._label, limit = ADABOOST_LIMIT)
                output, fp_num = self.strong_classifier[i].train()
                self.strong_classifier[i].saveModel(cache_filename)
                self.updateTrainingDate(output, fp_num)

            self.strong_classifier[i].makeClassifierPic()

            self.classifierNum += 1


    def updateTrainingDate(self, output, fp_num):

        assert len(output) == self._label.size

        mat = numpy.zeros((FEATURE_NUM, POSITIVE_SAMPLE + fp_num), dtype=numpy.float16)

        for i in xrange(POSITIVE_SAMPLE):
            for j in xrange(self.haar.featuresNum):
                mat[j][i] = self._mat[j][i]

        counter = 0
        # only reserve negative samples which are classified wrong
        for i in xrange(POSITIVE_SAMPLE, self._label.size):
            if output[i] != self._label[i]:
                for j in xrange(self.haar.featuresNum):
                    mat[j][POSITIVE_SAMPLE + counter] = self._mat[j][i]
                counter += 1

        self._mat = mat

    def predict(self):

        output = numpy.zeros(POSITIVE_SAMPLE + NEGATIVE_SAMPLE, dtype= numpy.float16)
        for i in xrange(self.classifierNum):

            self.strong_classifier[i].prediction(mat, th = 0)

            """unfinished"""

    def save(self):
        pass

    def is_goodenough(self):
        pass
