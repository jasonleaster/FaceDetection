#EFace -- A project of face detection

|Build Status|

working on :)
----

It's stimulating to do this project.
Enjoy with it.

During this period when I working on the project, I meet a lots of problem. But I also want to say "thanks" to these problem. It help me a lot to enhance my ability in programming.

* Exception Handle
    The training process cost too much time. Sometimes, we have a better idea to change the code into a better version. But the trainning process is going on. If we press `ctrl + c` to interrupt, the data that we have get from the `AdaBoost` process will lost.

    I use a handler for `KeyboardInterrupt` and then save the data of model so that the valuable data won't be lose.

* High Performance Programming in Python

    There have lots of tricks to make native Python code run more faster. The computation of image processing is very huge. This means that it's a typical problem about CPU-bound.

... ...

---
Optimization diary

2016-04-09 Restart to built this project and finished optimize the image.py

2016-04-13 refactor the training.py and make it more light. create a new module mapReduce.py. In haarFeature.py, @idxVector is initialized by numpy.zeros, it's faster than numpy.array([0 for i in range(length)])

2016-04-15 going to optimal weakClassifier.py and adaboost.py. Try to vectorize weakclassifier.py

2016-04-16 change scanImage.py and use different size of final classifier image but not resize the inputed image.
