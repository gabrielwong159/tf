# MNIST classification with Siamese networks
## About
Siamese CNNs work by comparing the outputs of different images from the same CNN, and determining whether or not the two images are similar based on the distances between the outputs. For more information on Siamese CNNs, check out my [other project on Siamese CNNs](https://github.com/gabrielwong159/siamese).

In this version of Siamese CNN, the final fully-connected layer has 2 neurons. The L1 loss between the outputs of two images is then combined with a softmax classifier to produce a binary classification of `0` (not similar) or `1` (similar).

## Getting started
Model definition is found in `model.py`. To change to L2 loss, change the logits computation from `tf.abs` to `tf.square`. Note that the last fully-connected layer has no activation function.

## Resources
Check out the [`References`](https://github.com/gabrielwong159/siamese#references) section of my other project.
