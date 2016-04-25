"""
Programmer  :   EOF
E-mail      :   jasonleaster@163.com
Date        :   2016.04.13
File        :   calFeature.py

"""
import numpy

from config import TRAINING_IMG_HEIGHT
from config import TRAINING_IMG_WIDTH

from haarFeature import Feature




def calFeature(Face, NonFace, features, FileName):
    """
    Parameter:
        @Face       : The face images set which is a instance of @ImageSet
        @NonFace    : The set of images which doesn't contain faces. 
                      It's also a instance of @ImageSet.

        @features   : A array or a list that contain features.
        @FileName   : A string which is the name of a file."""

    from config import HAAR_FEATURE_TYPE_I
    from config import HAAR_FEATURE_TYPE_II
    from config import HAAR_FEATURE_TYPE_III
    from config import HAAR_FEATURE_TYPE_IV

    haar = Feature(TRAINING_IMG_WIDTH, TRAINING_IMG_HEIGHT)

    faceSampleNum = Face.sampleNum
    tolSampleNum  = Face.sampleNum + NonFace.sampleNum
    featureNum    = len(features)

    Original_Data = numpy.zeros((featureNum, tolSampleNum))

    for f_dem in xrange(featureNum):
        (types, x, y, w, h) = features[f_dem]

        for i in xrange(Face.sampleNum):
            if types == HAAR_FEATURE_TYPE_I:
                Original_Data[f_dem][i] = haar.VecFeatureTypeI(   Face.images[i].vecImg, x, y, w, h)
            elif types == HAAR_FEATURE_TYPE_II:
                Original_Data[f_dem][i] = haar.VecFeatureTypeII(  Face.images[i].vecImg, x, y, w, h)
            elif types == HAAR_FEATURE_TYPE_III:
                Original_Data[f_dem][i] = haar.VecFeatureTypeIII( Face.images[i].vecImg, x, y, w, h)
            elif types == HAAR_FEATURE_TYPE_IV:
                Original_Data[f_dem][i] = haar.VecFeatureTypeIV(  Face.images[i].vecImg, x, y, w, h)

        for j in xrange(NonFace.sampleNum):
            if types == HAAR_FEATURE_TYPE_I:
                Original_Data[f_dem][j + faceSampleNum] = haar.VecFeatureTypeI(   NonFace.images[j].vecImg, x, y, w, h)
            elif types == HAAR_FEATURE_TYPE_II:
                Original_Data[f_dem][j + faceSampleNum] = haar.VecFeatureTypeII(  NonFace.images[j].vecImg, x, y, w, h)
            elif types == HAAR_FEATURE_TYPE_III:
                Original_Data[f_dem][j + faceSampleNum] = haar.VecFeatureTypeIII( NonFace.images[j].vecImg, x, y, w, h)
            elif types == HAAR_FEATURE_TYPE_IV:
                Original_Data[f_dem][j + faceSampleNum] = haar.VecFeatureTypeIV(  NonFace.images[j].vecImg, x, y, w, h)


    sync(Original_Data, FileName)

    return Original_Data
