# Computer vision with TensorFlow <img src="https://cdn-images-1.medium.com/max/256/1*cKG1LJvVTaWqSkYSyVqtsQ.png" width="40px" />
A collection of computer vision tasks on TensorFlow that encompass various implementations of CNN architectures. These serve as a reference and/or proof-of-concept for further adaptation.

## Projects
### `classification` - MNIST classifier
<p align="center">
    <img alt="MNIST example"
         src="https://www.tensorflow.org/images/mnist_0-9.png" />
</p>

Simple softmax classifier for MNIST digits. Architecture consists of two convolutional layers, followed by two fully-connected layers.

Computer vision tasks are often initially implemented on the MNIST dataset to ensure that the model is correctly defined, as well as to produce a baseline for model performance. As such, this serves as a guide to quickly set up an MNIST dataset, as well as reference for a simple baseline CNN architecture.

### `vgg` - VGG16 net
TensorFlow-Slim implementation of the VGG-16 architecture. VGG-16 is often used in transfer learning, in which the VGG-16 architecture is used to obtain a feature map, and the pretrained weights are loaded before training. Thus this serves as a quick reference to getting a VGG-16 network up with minimal effort.

<p align="center">
    <img alt="VGG-16 model architecture" 
         src="https://qph.fs.quoracdn.net/main-qimg-83c7dee9e8b039c3ca27c8dd91cacbb4" />
</p>

### `fcn` - Fully convolutional networks
Fully convolutional networks are typically used for semantic segmentation. The network architecture implemented here is from the paper titled `Fully convolutional networks for semantic segmentation`, in which VGG-16 is used as the base convolutional network.

<p align="center">
    <img alt="FCN-8s model architecture"
         src="https://devblogs.nvidia.com/wp-content/uploads/2016/11/figure15.png" />
</p>

The aim of the project is to help in the understanding of upsampling through transposed convolutional layers, and a U-net is now more commonly used as a baseline architecture for semantic segmentation tasks.

### `mtl` - Multi task learning
Quick proof-of-concept for multi task learning, in which two different tasks share the same base convolutional network. This shared feature map then branches into two fully-connected network components to produce two sets of outputs. In the example, we train the model to predict the digit and colour classification independently. The convolutional network is trained across all tasks through the summation of individual loss functions.

<p align="center">
    <img alt="MTL: classification & regression"
         src="https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/more_images/LocalizationRegression2.png" />
</p>

Such architectures are commonly used for object localization tasks, in which both a classification and a regression task are being performed on the same network.

### `siamese` - Siamese CNN
Siamese networks are often used in one-shot learning tasks, such as performing classification on images that do not appear in the training set.

Some implementations of Siamese CNNs compare the distance between the outputs of the last fully-connected layer across images, and perform classification based on the pair with the smallest distance. In this particular implementation, we extend the network to perform a binary softmax classification on a pair of images, to indicate whether these images are similar or not.

<p align="center">
    <img alt="Example Siamese CNN architecture"
         src="https://camo.githubusercontent.com/b27757e11d8687dc846b016e0fac80a544e7b645/68747470733a2f2f736f72656e626f756d612e6769746875622e696f2f696d616765732f5369616d6573655f6469616772616d5f322e706e67" />
</p>