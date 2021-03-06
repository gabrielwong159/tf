# Computer vision with TensorFlow <img src="https://cdn-images-1.medium.com/max/256/1*cKG1LJvVTaWqSkYSyVqtsQ.png" width="40px" />
A collection of computer vision tasks on TensorFlow that encompass various implementations of CNN architectures. These serve as a reference and/or proof-of-concept for further adaptation. In addition, being able to explain some of the concepts that I have applied helps me to consolidate my understanding.

> If you can't explain it simply, you don't understand it well enough.

## List of projects
1. [`mnist`: MNIST classifier](#mnist)
1. [`vgg`: VGG-16 net](#vgg)
1. [`fcn`: Fully convolutional networks](#fcn)
1. [`siamese`: Siamese CNNs](#siamese)
1. [`vae`: Variational autoencoder](#vae)
1. [`gan`: Generative adversarial networks](#gan)
1. [`mtl`: Multi-task learning](#mtl)
1. [`summary`: TensorFlow SummaryWriter and TensorBoard](#summary)
1. [`roi_pool`: ROI pooling](#roi_pool)


## Project details
### [`mnist`](./mnist/)
**MNIST classifier**  
<p align="center">
    <img alt="MNIST example"
         src="https://www.tensorflow.org/images/mnist_0-9.png" />
</p>

Simple softmax classifier for MNIST digits. Architecture consists of two convolutional layers, followed by two fully-connected layers.

Computer vision tasks are often initially implemented on the MNIST dataset to ensure that the model is correctly defined, as well as to produce a baseline for model performance. As such, this serves as a guide to quickly set up an MNIST dataset, as well as reference for a simple baseline CNN architecture.

### [`vgg` ](./vgg/)
**VGG-16 net**  
TensorFlow-Slim implementation of the VGG-16 architecture. VGG-16 is often used in transfer learning, in which the VGG-16 architecture is used to obtain a feature map, and the pretrained weights are loaded before training. Thus this serves as a quick reference to getting a VGG-16 network up with minimal effort.

<p align="center">
    <img alt="VGG-16 model architecture" 
         src="https://qph.fs.quoracdn.net/main-qimg-83c7dee9e8b039c3ca27c8dd91cacbb4" />
</p>

### [`fcn`](./fcn/)
**Fully convolutional networks**  
Fully convolutional networks are typically used for _semantic segmentation_. The network architecture implemented here is from the paper titled [_Fully convolutional networks for semantic segmentation_](https://arxiv.org/abs/1411.4038), in which _VGG-16_ is used as the base convolutional network.

<p align="center">
    <img alt="FCN-8s model architecture"
         src="https://devblogs.nvidia.com/wp-content/uploads/2016/11/figure15.png" />
</p>

The aim of the project is to help in the understanding of upsampling through transposed convolutional layers, and a _U-net_ is now more commonly used as a baseline architecture for semantic segmentation tasks.

### [`siamese`](./siamese/)
**Siamese CNNs**  
Siamese networks are often used in _one-shot learning_ tasks, such as performing classification on images that do not appear in the training set.

Some implementations of Siamese CNNs compare the distance between the outputs of the last fully-connected layer across images, and perform a binary classification based on the distance. If the distance between the two images is sufficiently small, then we take those two images to be similar.

<p align="center">
    <img alt="Example Siamese CNN architecture"
         src="https://camo.githubusercontent.com/b27757e11d8687dc846b016e0fac80a544e7b645/68747470733a2f2f736f72656e626f756d612e6769746875622e696f2f696d616765732f5369616d6573655f6469616772616d5f322e706e67" />
</p>

### [`vae`](./vae/)
**Variational autoencoders**  
A sample model architecture for a varational autoencoder trained on the MNIST dataset. The model is split into two components, the _encoder_ and the _decoder_. In this example, the two networks are decoupled such that we can feed values directly into the decoder network independently of the encoder network.

[VAE demo](https://transcranial.github.io/keras-js/#/mnist-vae)

<p align="center">
    <img alt="Illustration of VAE results"
         src="https://docs.pymc.io/_images/notebooks_convolutional_vae_keras_advi_36_0.png" />
</p>

### [`gan`](./gan/)
**Generative adversarial networks**  
Sample GAN for generating MNIST images. The model architecture consists of a _generator_, which takes in a random input vector and produces an image, and a _discriminator_, which attempts to distinguish between genuine and generated images.

While GANs generally take longer to train, as compared to VAEs, the images produced are of a higher quality. GANs are thus the preferred network for synthesizing images.

<p align="center">
	<img alt="Overall architecture of GANs"
		 src="https://skymind.ai/images/wiki/GANs.png" />
</p>

### [`mtl`](./mtl/)
**Multi-task learning**  
Quick proof-of-concept for multi-task learning, in which two different tasks share the same base convolutional network. This shared feature map then branches into two fully-connected network components to produce two sets of outputs. In the example, we train the model to predict the digit and colour classification independently. The convolutional network is trained across all tasks through the summation of individual loss functions.

<p align="center">
    <img alt="MTL: classification & regression"
         src="https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/more_images/LocalizationRegression2.png" />
</p>

Such architectures are commonly used for object localization tasks, in which both a classification and a regression task are being performed on the same network.

### [`summary`](./summary/)
**TensorFlow SummaryWriter and TensorBoard**  
Reference code for using `tf.summary` methods, for visualization using TensorBoard. Some useful information that can be visualized are the model loss and accuracy over time, or the computation graph of the model itself. This can be helpful when running multiple experiments across various hyperparameters, such that you can easily [compare their performances in TensorBoard](https://github.com/tensorflow/tensorboard/blob/master/README.md#runs-comparing-different-executions-of-your-model). 

### [`roi_pool`](./roi_pool/)
**Region of interest (ROI) pooling**  
ROI pooling was used in [Fast R-CNN](https://arxiv.org/abs/1504.08083) to convert variably-sized crops of feature maps into a fixed size. This allows us to perform tasks such as regression and classification on input images of different sizes, since fully-connected layers require inputs of specific sizes.

![](https://cdn-sv1.deepsense.ai/wp-content/uploads/2017/02/roi_pooling-1.gif)