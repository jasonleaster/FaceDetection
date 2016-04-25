"""
Programmer  :   EOF
File        :   init_training_set.py
Date        :   2015.12.29
E-mail      :   jasonleaster@163.com

Description :
    This script file will initialize the image set
and read all images in the directory which is given by
user.

"""
import numpy
import os
import pylab

from matplotlib import pyplot
from matplotlib import image

class Image:
    def __init__(self, fileName = None, label = None, Mat = None):
        if fileName != None:
            self.imgName = fileName
            self.img     = image.imread(fileName)

            if len(self.img.shape) == 3:
                self.img     = self.img[:,:, 1]

        else:
            assert Mat != None
            self.img     = Mat

        self.label   = label

        self.stdImg  = Image._normalization(self.img)

        self.iimg    = Image._integrateImg(self.stdImg)

        self.vecImg  = self.iimg.transpose().flatten()

    @staticmethod
    def _integrateImg(image):

        assert image.__class__ == numpy.ndarray

        row, col = image.shape
        #@iImg is integrated image of normalized image @self.stdImg
        iImg = numpy.zeros((row, col))

        for i in xrange(0, row):
            for j in xrange(0, col):
                if j == 0:
                    iImg[i][j] = image[i][j]
                else:
                    iImg[i][j] = iImg[i][j - 1] + image[i][j]

        for j in xrange(0, col):
            for i in xrange(1, row):
                iImg[i][j] += iImg[i - 1][j]

        return iImg

    @staticmethod
    def _normalization(image):

        assert image.__class__ == numpy.ndarray

        row, col = image.shape

        #stdImag standardized image
        stdImg = numpy.zeros((row, col))
        """
            What image.sum() do is the same as the following code 
        but more faster than this.

        for i in xrange(self.Row):
            for j in xrange(self.Col):
                sigma += image[i][j]
        """
        #sigma = image.sum()

        meanVal = image.mean()

        stdValue = image.std()
        if stdValue == 0:
            stdValue = 1

        stdImg = (image - meanVal) / stdValue

        return stdImg

    @staticmethod
    def show(image = None):
        if image == None:
            return
        pyplot.matshow(image)
        pylab.show()


class ImageSet:
    def __init__(self, imgDir = None, label = None, sampleNum = None):

        assert isinstance(imgDir, str)

        self.imgDir = imgDir
        self.fileList = os.listdir(imgDir)
        self.fileList.sort()

        if sampleNum == None:
            self.sampleNum = len(self.fileList)
        else:
            self.sampleNum = sampleNum

        self.curFileIdx = self.sampleNum
        self.label  = label

        self.images = [None for _ in xrange(self.sampleNum)]

        processed = -10.
        for i in xrange(self.sampleNum):
            self.images[i] = Image(imgDir + self.fileList[i], label)

            if i % (self.sampleNum / 10) == 0:
                processed += 10.
                print "Loading ", processed, "%"

        print "Loading  100 %\n"

    def readNextImg(self):
        img = Image(self.imgDir + self.fileList[self.curFileIdx], self.label)
        self.curFileIdx += 1
        return img
