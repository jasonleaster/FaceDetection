"""
Just a helper script
"""
from matplotlib import pyplot
import numpy
import os
import pylab

from image              import ImageSet
from config             import *
from adaboost           import AdaBoost
from haarFeature        import Feature
from getCachedAdaBoost  import getCachedAdaBoost

combined_ROWNUM = 10
combined_COLNUM = 10

TestSetFace = ImageSet(TEST_FACE, sampleNum = combined_COLNUM * combined_ROWNUM)

IMG_WIDTH  = TestSetFace.images[0].Row
IMG_HEIGHT = TestSetFace.images[0].Col

combinedImage = numpy.array([[0 for i in xrange(IMG_WIDTH * combined_COLNUM)]
                                for j in xrange(IMG_HEIGHT* combined_ROWNUM)])

for row in xrange(combined_ROWNUM):
    for col in xrange(combined_COLNUM):
        #row * 10 + col == i
        row_start = row     * IMG_HEIGHT
        row_end   = (row+1) * IMG_HEIGHT
        col_start = col     * IMG_WIDTH
        col_end   = (col+1) * IMG_WIDTH
        index     = row * combined_COLNUM + col
        combinedImage[row_start: row_end, col_start:col_end] = TestSetFace.images[index].img[:]
pyplot.axis('off')
pyplot.imshow(combinedImage, cmap = "gray", interpolation = 'nearest')
pyplot.savefig("./figure/combinedImage.jpg", bbox_inches = "tight", pad_inches = 0)
#pylab.show()
