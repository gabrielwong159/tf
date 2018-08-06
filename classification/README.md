# MNIST classification
## About
> When one learns how to program, there's a tradition that the first thing you do is print "Hello World."
> Just like programming has Hello World, machine learning has MNIST.

This project serves as a simple guide to the OOP approach to organizing Tensorflow projects. This allows the user to easily access variables and placeholders during model training. In addition, `TensorFlow-Slim` is used as opposed to traditional TensorFlow code.

## Getting started
The model definition can be found in `model.py`.

![Model architecture](https://cdn-images-1.medium.com/max/800/1*cPAmSB9nziZPI73VC5HAHg.png)

Training and testing are in `train.py` and `test.py` respectively. The model currently trains for 5000 iterations, attaining an accuracy of `99.0%`. Increasing the number of iterations to 20000 should yield `~99.2%` accuracy.

## Resources
### TensorFlow tutorials
The code here draws inspiration from the `GET STARTED` section of the TensorFlow website. These serve as a good starting point for familiarizing yourself with TensorFlow.

* [MNIST for ML Beginners](https://www.tensorflow.org/versions/r1.0/get_started/mnist/beginners)
* [Deep MNIST for Experts](https://www.tensorflow.org/versions/r1.0/get_started/mnist/pros)

### [Tensorflow: The Confusing Parts (1)](https://jacobbuckman.com/post/tensorflow-the-confusing-parts-1/)
This is a particularly useful article when beginning TensorFlow, which helps the reader to understand the differences in TensorFlow better. Highly recommended read when starting out in TensorFlow.

### [TensorFlow-Slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim)
TF-Slim helps simplify code vastly when creating models in TensorFlow.
