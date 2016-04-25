## E-Face

This project name as `E-Face` which is a implementation of face detection algorithm.

My nick name is `EOF`. For convenient, I name it as `E-Face`.

### The archtecture of this project.

The following list show the files in this awesome project.

* adaboost.py 
Implmentation of Adaptive Boosting algorithm

* cascade.py
Cascade Decision Tree

* config.py
All parameters of configuration in this project are stored in this file.

* image.py
The initialization of images. class Image and class ImageSet are in this file.

* haarFeature.py
Stuff with Haar-Features.

* vecProduct.py
A simple function to do production of two vectors.

* weakClassifier.py
The detail about Weak classifier.

* testing.py
Script for testing.

* training.py
Script for training the model.

* getCachedAdaBoost.py

directories:

* model/
    cache files for adaboost model.

* featuers/
    values for different feaures with different samples.

* doc/
    documents with this project.

###Programming Style:

    I used basic OOP(Object Oriented Programming) tricks to build my program. Something like... I put all about `AdaBoost` into a class(AdaBoost) which you can find in file `adaboost.py`. Everytime you want to do something with adaboost, just create a object instance of that class.

Adavantages of this style:
    Higher level of abstraction and easy to be used. With this style, green hand will easy to build good archtecture with our project.

Disadvantages of this style:
    Without optimalization, it will cost a lot of memory. This will be obvious when the scale of project goes more and more large.



