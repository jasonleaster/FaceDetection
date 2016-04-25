"""
Programmer  :   EOF
File        :   detector.py
E-mail      :   jasonleaster@163.com
Date        :   2016.01.18
"""

from config         import *
from image          import Image
from haarFeature    import Feature
from matplotlib     import pyplot
from adaboost       import getCachedAdaBoost
import pylab
import numpy

def scanImgAtScale(model, image, scale):
    assert isinstance(image, numpy.ndarray)

    ImgHeight, ImgWidth = image.shape

    SEARCH_WIN_WIDTH  = int(TRAINING_IMG_WIDTH  * scale)
    SEARCH_WIN_HEIGHT = int(TRAINING_IMG_HEIGHT * scale)

    subImages = []
    subWins   = []
    for x in xrange(0, ImgWidth - SEARCH_WIN_WIDTH - 10, SEARCH_WIN_WIDTH/4):
        for y in xrange(0, ImgHeight - SEARCH_WIN_HEIGHT - 10, SEARCH_WIN_HEIGHT/4):
            searchWindow = (x, y, SEARCH_WIN_WIDTH, SEARCH_WIN_HEIGHT)
            subWins.append(searchWindow)
            subImages.append( Image(Mat = image[y:y+SEARCH_WIN_HEIGHT, x:x+SEARCH_WIN_WIDTH]))

    subImgNum = len(subImages)


    selFeatures = []

    haar_scaled = Feature(SEARCH_WIN_WIDTH,   SEARCH_WIN_HEIGHT)
    haar_train  = Feature(TRAINING_IMG_WIDTH, TRAINING_IMG_HEIGHT)

    for n in xrange(model.N):
        selFeatures.append(haar_train.features[ model.G[n].opt_dimension ] + tuple([model.G[n].opt_dimension]))

    mat = numpy.zeros((FEATURE_NUM, subImgNum), dtype=numpy.float16)

    for feature in selFeatures:
        (types, x, y, w, h, dim) = feature

        x = int(x * scale)
        y = int(y * scale)
        w = int(w * scale)
        h = int(h * scale)

        for i in xrange(len(subImages)):
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
        #if output[i] == LABEL_POSITIVE:
        if output[i] > AB_TH:
            candidate = numpy.array(subWins[i])
            x, y, w, h = candidate
            rectangle.append((x, y, w, h, output[i]))

    return rectangle

def scanImgOverScale(image):

    from config import DETECT_START
    from config import DETECT_END
    from config import DETECT_STEP

    model = getCachedAdaBoost(filename = ADABOOST_CACHE_FILE + str(0), limit = ADABOOST_LIMIT)

    rectangles = []

    for scale in numpy.arange(DETECT_START , DETECT_END, DETECT_STEP):
        rectangles += scanImgAtScale(model, image, scale)

    return rectangles

def drawRectangle(image, x, y, width, height):
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

def optimalRectangle(rectangles, overlap_threshold):

    # number of rectangles
    numRec = len(rectangles)

    for i in xrange(numRec):
        for j in xrange(i+1, numRec):
            if rectangles[i][4] < rectangles[j][4]:
                rectangles[i], rectangles[j] = \
                tuple(rectangles[j]), tuple(rectangles[i])

    overlapRate = [[None for i in xrange(numRec)]
                         for j in xrange(numRec)]
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
    for i in xrange(numRec):
        x1, y1, w1, h1, score1 = rectangles[i]
        area_1 = w1 * h1
        for j in xrange(i, numRec):
            x2, y2, w2, h2, score2 = rectangles[j]
            area_2 = h2 * w2

            leftx = max(x1, x2) * 1.
            lefty = min(y1+h1, y2+h2) * 1.
            rightx = min(x1 + w1, x2 + w2) * 1.
            righty = max(y1, y2) * 1.

            if leftx > rightx or lefty > righty:
                overlapRate[i][i] = 0
            else:
                overlapRate[i][j] = ((rightx - leftx)*(righty - lefty)) / (area_1 + area_2)

    reduced = [i for i in xrange(numRec)]
    for i in xrange(numRec):
        for j in xrange(i + 1, numRec):
            if overlapRate[i][j] > overlap_threshold:
                reduced[j] = reduced[i]

    reducedRectangels = []
    for i in numpy.unique(reduced):
        reducedRectangels.append(rectangles[i])

    return reducedRectangels

from matplotlib import image

img = image.imread(TEST_IMG)

img = img[:,:, 1]

rectangles = scanImgOverScale(img)

rectangles = optimalRectangle(rectangles, overlap_threshold = OVER_LAP_TH)

# ------- for debug ------------
HEIGHT, WIDTH = img.shape

print rectangles
print "number of rectangles:", len(rectangles)

img = image.imread(TEST_IMG)
for rectangle in rectangles:
    x, y, width, height, score = rectangle

    drawRectangle(img, x, y, width, height)
# -------------------------------
