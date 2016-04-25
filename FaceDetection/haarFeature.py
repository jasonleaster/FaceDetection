"""
Programmer  :   EOF
Date        :   2016.01.16
E-mail      :   jasonleaster@163.com
File        :   haarFeature.py

Description:

    Types of Haar-like rectangle features
     --- ---
    |   +   |
    |-------|
    |   -   |
     -------
        I
     --- ---     
    |   |   |    
    | - | + |    
    |   |   |    
     --- ---     
       II

     -- -- -- 
    |  |  |  |
    |- | +| -|
    |  |  |  |
     -- -- -- 
       III

     --- ---
    | - | + |
    |___|___|
    | + | - |
    |___|___|
       IV

    For each feature pattern, the start point(x, y) is at 
    the most left-up pixel in that window. The size of that
    window is @width * @height
"""
import numpy

from config import HAAR_FEATURE_TYPE_I
from config import HAAR_FEATURE_TYPE_II
from config import HAAR_FEATURE_TYPE_III
from config import HAAR_FEATURE_TYPE_IV

from image import Image

class Feature:
    def __init__(self, img_Width, img_Height):

        self.featureName = "Haar Feature"

        self.img_Width  = img_Width
        self.img_Height = img_Height

        self.tot_pixels = img_Width * img_Height

        self.featureTypes = (HAAR_FEATURE_TYPE_I,
                             HAAR_FEATURE_TYPE_II,
                             HAAR_FEATURE_TYPE_III,
                             HAAR_FEATURE_TYPE_IV)

        self.features    = self._evalFeatures_total();

        self.featuresNum = len(self.features)

        #self.featureMat  =     numpy.zeros((self.tot_pixels, self.featuresNum),
        #                                   dtype=numpy.float16)

        # just for running faster and save RAM. allocate once and use many times.
        self.vector          = numpy.zeros(self.featuresNum, dtype=numpy.float16)

        self.idxVector_tmp_0 = numpy.zeros(self.tot_pixels, dtype = numpy.int)
        self.idxVector_tmp_1 = numpy.zeros(self.tot_pixels, dtype = numpy.int)
        self.idxVector_tmp_2 = numpy.zeros(self.tot_pixels, dtype = numpy.int)
        self.idxVector_tmp_3 = numpy.zeros(self.tot_pixels, dtype = numpy.int)


    def vecRectSum(self, idxVector, x, y, width, height):
        idxVector *= 0 # reset this vector
        if x == 0 and y == 0:
            idxVector[width * height + 2] = +1

        elif x == 0:
            idx1 = self.img_Height * (    width - 1) + height + y - 1
            idx2 = self.img_Height * (    width - 1) +          y - 1
            idxVector[idx1] = +1
            idxVector[idx2] = -1

        elif y == 0:
            idx1 = self.img_Height * (x + width - 1) + height - 1
            idx2 = self.img_Height * (x         - 1) + height - 1
            idxVector[idx1] = +1
            idxVector[idx2] = -1
        else:
            idx1 = self.img_Height * (x + width - 1) + height + y - 1
            idx2 = self.img_Height * (x + width - 1) +          y - 1
            idx3 = self.img_Height * (x         - 1) + height + y - 1
            idx4 = self.img_Height * (x         - 1) +          y - 1

            assert idx1 < self.tot_pixels and idx2 < self.tot_pixels 
            assert idx3 < self.tot_pixels and idx4 < self.tot_pixels 

            idxVector[idx1] = + 1
            idxVector[idx2] = - 1
            idxVector[idx3] = - 1
            idxVector[idx4] = + 1

        return idxVector


    def VecFeatureTypeI(self, vecImg, x, y, width, height):
        vec1 = self.vecRectSum(self.idxVector_tmp_0, x, y         , width, height)
        vec2 = self.vecRectSum(self.idxVector_tmp_1, x, y + height, width, height)

        featureSize = width * height * 2

        return (vec1.dot(vecImg) - vec2.dot(vecImg))/featureSize

    def VecFeatureTypeII(self, vecImg, x, y, width, height):
        vec1 = self.vecRectSum(self.idxVector_tmp_0, x + width, y, width, height)
        vec2 = self.vecRectSum(self.idxVector_tmp_1, x        , y, width, height)

        featureSize = width * height * 2

        return (vec1.dot(vecImg) - vec2.dot(vecImg))/featureSize

    def VecFeatureTypeIII(self,vecImg, x, y, width, height):
        vec1 = self.vecRectSum(self.idxVector_tmp_0, x +   width, y, width, height)
        vec2 = self.vecRectSum(self.idxVector_tmp_1, x          , y, width, height)
        vec3 = self.vecRectSum(self.idxVector_tmp_2, x + 2*width, y, width, height)

        featureSize = width * height * 3

        return (vec1.dot(vecImg) - vec2.dot(vecImg)
                - vec3.dot(vecImg))/featureSize

    def VecFeatureTypeIV(self, vecImg, x, y, width, height):
        vec1 = self.vecRectSum(self.idxVector_tmp_0, x + width,          y, width, height)
        vec2 = self.vecRectSum(self.idxVector_tmp_1, x        ,          y, width, height)
        vec3 = self.vecRectSum(self.idxVector_tmp_2, x        , y + height, width, height)
        vec4 = self.vecRectSum(self.idxVector_tmp_3, x + width, y + height, width, height)

        featureSize = width * height * 4

        return (vec1.dot(vecImg) - vec2.dot(vecImg) +
                vec3.dot(vecImg) - vec4.dot(vecImg))/featureSize

    def _evalFeatures_total(self):
        win_Height = self.img_Height
        win_Width  = self.img_Width

        height_Limit = {HAAR_FEATURE_TYPE_I   : win_Height/2 - 1,
                         HAAR_FEATURE_TYPE_II  : win_Height   - 1,
                         HAAR_FEATURE_TYPE_III : win_Height   - 1,
                         HAAR_FEATURE_TYPE_IV  : win_Height/2 - 1}

        width_Limit  = {HAAR_FEATURE_TYPE_I   : win_Width   - 1,
                        HAAR_FEATURE_TYPE_II  : win_Width/2 - 1,
                        HAAR_FEATURE_TYPE_III : win_Width/3 - 1,
                        HAAR_FEATURE_TYPE_IV  : win_Width/2 - 1}

        features = []
        for types in self.featureTypes:
            for w in xrange(1, width_Limit[types]):
                for h in xrange(1, height_Limit[types]):

                    if types == HAAR_FEATURE_TYPE_I:
                        x_limit = win_Width  - w
                        y_limit = win_Height - 2*h
                        for x in xrange(1, x_limit):
                            for y in xrange(1, y_limit):
                                features.append( (types, x, y, w, h))

                    elif types == HAAR_FEATURE_TYPE_II:
                        x_limit = win_Width  - 2*w
                        y_limit = win_Height - h
                        for x in xrange(1, x_limit):
                            for y in xrange(1, y_limit):
                                features.append( (types, x, y, w, h))

                    elif types == HAAR_FEATURE_TYPE_III:
                        x_limit = win_Width  - 3*w
                        y_limit = win_Height - h
                        for x in xrange(1, x_limit):
                            for y in xrange(1, y_limit):
                                features.append( (types, x, y, w, h))

                    elif types == HAAR_FEATURE_TYPE_IV:
                        x_limit = win_Width  - 2*w
                        y_limit = win_Height - 2*h
                        for x in xrange(1, x_limit):
                            for y in xrange(1, y_limit):
                                features.append( (types, x, y, w, h))

        return features


    def _evalFeatures(self):
        win_Height = self.img_Height
        win_Width  = self.img_Width

        height_Limit = {HAAR_FEATURE_TYPE_I   : win_Height/2 - 1,
                        HAAR_FEATURE_TYPE_II  : win_Height   - 1,
                        HAAR_FEATURE_TYPE_III : win_Height   - 1,
                        HAAR_FEATURE_TYPE_IV  : win_Height/2 - 1}

        width_Limit  = {HAAR_FEATURE_TYPE_I   : win_Width   - 1,
                        HAAR_FEATURE_TYPE_II  : win_Width/2 - 1,
                        HAAR_FEATURE_TYPE_III : win_Width/3 - 1,
                        HAAR_FEATURE_TYPE_IV  : win_Width/2 - 1}

        features = []
        for types in self.featureTypes:
            for w in xrange(1, width_Limit[types]):
                for h in xrange(1, height_Limit[types]):

                    y_start = None

                    if types == HAAR_FEATURE_TYPE_I:
                        x_limit = win_Width  - w
                        y_limit = win_Height - 2*h 
                        for x in xrange(1, x_limit):
                            for y in xrange(1, y_limit, 2):
                                features.append( (types, x, y, w, h))

                    elif types == HAAR_FEATURE_TYPE_II:
                        x_limit = win_Width  - 2*w  
                        y_limit = win_Height - h
                        for x in xrange(1, x_limit):
                            if h % 2 == 1:
                                if x % 2 == 1:
                                    y_start = 1
                                else:
                                    y_start = 2
                            else:
                                y_start = 1

                            for y in xrange(y_start, y_limit, 2):
                                features.append( (types, x, y, w, h))

                    elif types == HAAR_FEATURE_TYPE_III:
                        x_limit = win_Width  - 3*w  
                        y_limit = win_Height - h
                        for x in xrange(1, x_limit):
                            if w == 1:
                                if h % 2 == 1:
                                    if (h + 1)/2 % 2 == 1:
                                        if x % 2 == 1:
                                            y_start = 1
                                        else:
                                            y_start = 2
                                    else:
                                        if x % 2 == 1:
                                            y_start = 2
                                        else:
                                            y_start = 1
                                else:
                                    if (h/2) % 2 == 1:
                                        y_start = 2
                                    else:
                                        y_start = 1
                            elif w == 2:
                                if h % 2 == 1:
                                    if x % 2 == 1:
                                        y_start = 2
                                    else:
                                        y_start = 1
                                else:
                                    y_start = 2

                            elif w == 3:
                                if h % 2 == 1:
                                    if (h+1)/2 % 2 == 1:
                                        if x % 2 == 1:
                                            y_start = 2
                                        else:
                                            y_start = 1
                                    else:
                                        if x % 2 == 1:
                                            y_start = 1
                                        else:
                                            y_start = 2
                                else:
                                    if (h/2) % 2 == 1:
                                        y_start = 1
                                    else:
                                        y_start = 2
                            #elif w == 4:
                            else:
                                if h % 2 == 1:
                                    if x % 2 == 1:
                                        y_start = 1
                                    else:
                                        y_start = 2
                                else:
                                    y_start = 1

                            for y in xrange(y_start, y_limit, 2):
                                features.append( (types, x, y, w, h))

                    elif types == HAAR_FEATURE_TYPE_IV:
                        x_limit = win_Width  - 2*w 
                        y_limit = win_Height - 2*h
                        for x in xrange(1, x_limit):
                            for y in xrange(1, y_limit, 2):
                                features.append( (types, x, y, w, h))
        return features


    def calFeatureForImg(self, img):

        assert isinstance(img, Image)
        assert img.img.shape[0] == self.img_Height
        assert img.img.shape[1] == self.img_Width

        for i in xrange(self.featuresNum):
            type, x, y, w, h = self.features[i]

            if   type == HAAR_FEATURE_TYPE_I:
                self.vector[i] = self.VecFeatureTypeI(img.vecImg, x, y, w, h)
            elif type == HAAR_FEATURE_TYPE_II:
                self.vector[i] = self.VecFeatureTypeII(img.vecImg, x, y, w, h)
            elif type == HAAR_FEATURE_TYPE_III:
                self.vector[i] = self.VecFeatureTypeIII(img.vecImg, x, y, w, h)
            elif type == HAAR_FEATURE_TYPE_IV:
                self.vector[i] = self.VecFeatureTypeIV(img.vecImg, x, y, w, h)
            else:
                raise Exception("unknown feature type")

        return self.vector

    def makeFeaturePic(self, feature):

        from matplotlib import pyplot
        from config     import BLACK
        from config     import WHITE
        import pylab

        (types, x, y, width, height) = feature

        assert x >= 0 and x < self.img_Width
        assert y >= 0 and y < self.img_Height
        assert width > 0 and height > 0

        image = numpy.array([[125. for i in xrange(self.img_Width)]
                                 for j in xrange(self.img_Height)])

        if types == HAAR_FEATURE_TYPE_I:
            for i in xrange(y, y + height * 2):
                for j in xrange(x, x + width):
                    if i < y + height:
                        image[i][j] = WHITE
                    else:
                        image[i][j] = BLACK

        elif types == HAAR_FEATURE_TYPE_II:
            for i in xrange(y, y + height):
                for j in xrange(x, x + width * 2):
                    if j < x + width:
                        image[i][j] = BLACK
                    else:
                        image[i][j] = WHITE

        elif types == HAAR_FEATURE_TYPE_III:
            for i in xrange(y, y + height):
                for j in xrange(x, x + width * 3):
                    if j >= (x + width) and j < (x + width * 2):
                        image[i][j] = WHITE
                    else:
                        image[i][j] = BLACK

        elif types == HAAR_FEATURE_TYPE_IV:
            for i in xrange(y, y + height * 2):
                for j in xrange(x, x + width * 2):
                    if (j < x + width and i < y + height) or\
                       (j >= x + width and i >= y + height):
                        image[i][j] = BLACK
                    else:
                        image[i][j] = WHITE


        pyplot.matshow(image, cmap = "gray")
        pylab.show()

