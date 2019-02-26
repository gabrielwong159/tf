# VGG network
The VGG-16 network is often used in transfer learning. Here we provide an implementation of the VGG-16 network using TensorFlow-Slim. The network is modelled after the implementation given by TensorFlow.

<div align="center">
    <img
        src="https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/05/01000913/i3.png"
        alt="VGG-16 output shapes"
        width="300" />
    <img
        src="https://blog.keras.io/img/imgclf/vgg16_modified.png"
        alt="VGG-16 architecture overview"
        width="300" />
</div>

## Resources
* [Original TF VGG implementation](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/vgg.py)
* [Specifying the VGG16 Layers (TF-Slim)](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim#working-example-specifying-the-vgg16-layers)
* [`vgg_16.ckpt` download](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models)