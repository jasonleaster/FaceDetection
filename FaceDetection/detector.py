"""
Programmer  :   EOF
File        :   detector.py
E-mail      :   jasonleaster@163.com
Date        :   2016.01.18
"""

from config         import TRAINING_IMG_WIDTH
from config         import TRAINING_IMG_HEIGHT
from config         import HAAR_FEATURE_TYPE_I
from config         import HAAR_FEATURE_TYPE_II
from config         import HAAR_FEATURE_TYPE_III
from config         import HAAR_FEATURE_TYPE_IV
from config         import AB_TH
from config         import SEARCH_WIN_STEP

from image          import Image
from haarFeature    import Feature
from matplotlib     import pyplot
from adaboost       import getCachedAdaBoost
import pylab
import numpy


class Detector:

    def __init__(self):
        pass


    def scanImgAtScale(self, model, image, scale):
        assert isinstance(image, numpy.ndarray)

        ImgHeight, ImgWidth = image.shape

        SEARCH_WIN_WIDTH  = int(TRAINING_IMG_WIDTH  * scale)
        SEARCH_WIN_HEIGHT = int(TRAINING_IMG_HEIGHT * scale)

        width     = ImgWidth - SEARCH_WIN_WIDTH - 10
        height    = ImgHeight - SEARCH_WIN_HEIGHT - 10

        step      = SEARCH_WIN_WIDTH/SEARCH_WIN_STEP

        subWinNum = (width/step + 1) * (height/step + 1)

        subImages = numpy.zeros(subWinNum, dtype = object)
        subWins   = numpy.zeros(subWinNum, dtype = object)

        idx = 0
        for x in xrange(0, width, step):
            for y in xrange(0, height, step):
                subWins[idx]   = (x, y, SEARCH_WIN_WIDTH, SEARCH_WIN_HEIGHT)

                subImages[idx] = Image(Mat = image[y:y+SEARCH_WIN_HEIGHT, x:x+SEARCH_WIN_WIDTH])
                idx += 1

        assert idx <= subWinNum

        subImgNum = idx

        selFeatures = numpy.zeros(model.N, dtype=object)

        haar_scaled = Feature(SEARCH_WIN_WIDTH,   SEARCH_WIN_HEIGHT)
        haar_train  = Feature(TRAINING_IMG_WIDTH, TRAINING_IMG_HEIGHT)

        for n in xrange(model.N):
            selFeatures[n] = haar_train.features[ model.G[n].opt_dimension ] + tuple([model.G[n].opt_dimension])

        mat = numpy.zeros((haar_train.featuresNum, subImgNum), dtype=numpy.float16)

        for feature in selFeatures:
            (types, x, y, w, h, dim) = feature

            x = int(x * scale)
            y = int(y * scale)
            w = int(w * scale)
            h = int(h * scale)

            for i in xrange(subImgNum):
                if   types == HAAR_FEATURE_TYPE_I:
                    mat[dim][i] = haar_scaled.VecFeatureTypeI(subImages[i].vecImg, x, y, w, h)
                elif types == HAAR_FEATURE_TYPE_II:
                    mat[dim][i] = haar_scaled.VecFeatureTypeII(subImages[i].vecImg, x, y, w, h)
                elif types == HAAR_FEATURE_TYPE_III:
                    mat[dim][i] = haar_scaled.VecFeatureTypeIII(subImages[i].vecImg, x, y, w, h)
                elif types == HAAR_FEATURE_TYPE_IV:
                    mat[dim][i] = haar_scaled.VecFeatureTypeIV(subImages[i].vecImg, x, y, w, h)

        output = model.grade(mat)

        rectangle = []
        for i in xrange(len(output)):
            if output[i] > AB_TH:
                candidate = numpy.array(subWins[i])
                x, y, w, h = candidate
                rectangle.append((x, y, w, h, output[i]))

        return rectangle


    def scanImgOverScale(self, image):

        from config import DETECT_START
        from config import DETECT_END
        from config import DETECT_STEP
        from config import ADABOOST_CACHE_FILE
        from config import ADABOOST_LIMIT

        model = getCachedAdaBoost(filename = ADABOOST_CACHE_FILE + str(0), limit = ADABOOST_LIMIT)

        rectangles = []

        for scale in numpy.arange(DETECT_START , DETECT_END, DETECT_STEP):
            rectangles += self.scanImgAtScale(model, image, scale)

        return self.optimalRectangle(rectangles)


    def optimalRectangle(self, rectangles):

        # number of rectangles
        numRec = len(rectangles)

        for i in xrange(numRec):
            for j in xrange(i+1, numRec):
                if rectangles[i][4] < rectangles[j][4]:
                    rectangles[i], rectangles[j] = \
                    tuple(rectangles[j]), tuple(rectangles[i])

        """
        (x1, y1, w1, h1) represent as the first  rectangle.
        (x2, y2, w2, h2) represent as the second rectangle.
        |-------> x
        | _______
        ||  1  __|____
        ||____|__|    |
        |     |____2__|
        \/ y
        """
        reduced = [i for i in xrange(numRec)]
        for i in xrange(numRec):
            x1, y1, w1, h1, score1 = rectangles[i]
            area_1 = w1 * h1
            for j in xrange(i+1, numRec):
                x2, y2, w2, h2, score2 = rectangles[j]
                area_2 = h2 * w2

                if( self.pointInRectangle((x2,      y2     ), rectangles[i]) or
                    self.pointInRectangle((x2 + w2, y2     ), rectangles[i]) or
                    self.pointInRectangle((x2,      y2 + h2), rectangles[i]) or
                    self.pointInRectangle((x2 + w2, y2 + h2), rectangles[i]) or
                    self.pointInRectangle((x2 + w2/2,y2    ), rectangles[i]) or
                    self.pointInRectangle((x2      ,y2+h2/2), rectangles[i]) or
                    self.pointInRectangle((x2 + w2, y2+h2/2), rectangles[i]) or
                    self.pointInRectangle((x2 + w2/2,y2+h2 ), rectangles[i]) ) is True:

                    reduced[j] = reduced[i]

                if( self.pointInRectangle((x1,      y1     ), rectangles[j]) or
                    self.pointInRectangle((x1 + w1, y1     ), rectangles[j]) or
                    self.pointInRectangle((x1,      y1 + h1), rectangles[j]) or
                    self.pointInRectangle((x1 + w1, y1 + h1), rectangles[j]) or
                    self.pointInRectangle((x1 + w1/2,y1    ), rectangles[j]) or
                    self.pointInRectangle((x1      ,y1+h1/2), rectangles[j]) or
                    self.pointInRectangle((x1 + w1, y1+h1/2), rectangles[j]) or
                    self.pointInRectangle((x1 + w1/2,y1+h1 ), rectangles[j]) ) is True:

                    reduced[j] = reduced[i]

        reducedRectangels = []
        for i in numpy.unique(reduced):
            reducedRectangels.append(rectangles[i])

        return reducedRectangels


    def pointInRectangle(self, point, rectangle):
        m, n = point
        x, y, w, h, _ = rectangle

        if((x < m and m < x + w) and (y < n and n < y +h)):
            return True
        else:
            return False


    def drawRectangle(self, image, x, y, width, height):
        assert isinstance(image, numpy.ndarray)

        if len(image.shape) != 2:
            Row, Col, _ = image.shape
            if x + width >= Col or y + height >= Row:
                return
            image[y:y+height, x-1:x+1, 0] = 0
            image[y:y+height, x-1:x+1, 1] = 255
            image[y:y+height, x-1:x+1, 2] = 0

            image[y:y+height, (x-1 + width):(x+1 + width), 0] = 0
            image[y:y+height, (x-1 + width):(x+1 + width), 1] = 255
            image[y:y+height, (x-1 + width):(x+1 + width), 2] = 0

            image[y-1:y+1, x:x+width, 0] = 0
            image[y-1:y+1, x:x+width, 1] = 255
            image[y-1:y+1, x:x+width, 2] = 0

            image[(y-1+height):(y+height+1), x:x+width, 0] = 0
            image[(y-1+height):(y+height+1), x:x+width, 1] = 255
            image[(y-1+height):(y+height+1), x:x+width, 2] = 0
        else:
            Row, Col = image.shape
            if x + width >= Col or y + height >= Row:
                return

            image[y:y+height, x-1:x+1  ] = 255
            image[y:y+height, x-1 + width:x+1 + width] = 255
            image[y-1:y+1   , x:x+width] = 255
            image[y-1+height:y+1+height, x:x+width] = 255

        pyplot.imshow(image)
        pylab.show()


    def showResult(self, image, rectangles):

        for rectangle in rectangles:
            x, y, width, height, score = rectangle

            print rectangle
            self.drawRectangle(image, x, y, width, height)
